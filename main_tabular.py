import copy
import random
import numpy as np
import torch
import math
import time
import csv
import concurrent.futures
import yaml

from model import get_tabular_model
from data import (
    load_tabular_data,
    get_task_datasets,
    split_task_dataset_dirichlet,
    get_dataloader,
)
from train import train_one_epoch_tts
from federated import fedavg
from eval import evaluate
from buffer import ReplayBuffer

# ---------------- LOAD CONFIG ---------------- 
with open("config.yaml", "r") as f:
    config_yaml = yaml.safe_load(f)

cfg = config_yaml['tabular']
NUM_CLIENTS = cfg['num_clients']
NUM_ROUNDS_PER_TASK = cfg['num_rounds_per_task']


def get_task_cumulative_classes(task_classes):
    cumulative = {}
    total = 0
    for task_id in sorted(task_classes.keys()):
        total += len(task_classes[task_id])
        cumulative[task_id] = total
    return cumulative

def print_experiment_config(config_buffer_type, config_use_tts, method_name, num_tasks, dirichlet_alpha, task_classes):
    print("=" * 50)
    print(" " * 10 + "EXPERIMENT CONFIGURATIONS")
    print("=" * 50)
    print(f"Dataset                              : {cfg['dataset']} (Tabular)")
    print(f"Number of Tasks                      : {num_tasks}")
    print(f"Method Name                          : {method_name}")
    print(f"Buffer Strategy (CONFIG_BUFFER_TYPE) : {config_buffer_type}")
    print(f"Use TTS (CONFIG_USE_TTS)             : {config_use_tts}")
    print(f"Number of Clients (NUM_CLIENTS)      : {NUM_CLIENTS}")
    print(f"Rounds per Task (NUM_ROUNDS_PER_TASK): {NUM_ROUNDS_PER_TASK}")
    print(f"Dirichlet Alpha (DIRICHLET_ALPHA)    : {dirichlet_alpha}")
    print(f"Task Classes (TASK_CLASSES)          : {task_classes}")
    print("=" * 50 + "\n")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_single_client(client_id, client_loader, global_model, task_id, device, replay_buffer, num_old_classes, num_total_seen_classes, use_tts, current_lr):
    print(f"Start training Client{client_id} Task{task_id}")
    
    if client_loader is None or len(client_loader.dataset) == 0:
        print(f"Skipping Client{client_id} due to 0 samples.")
        return copy.deepcopy(global_model).to(device)

    local_model = copy.deepcopy(global_model).to(device)

    client_replay = replay_buffer.get_client_replay_data(client_id) if replay_buffer else None

    acc = train_one_epoch_tts(
        local_model,
        client_loader,
        device,
        replay_dataset=client_replay,
        batch_size=64,
        task_id=task_id,
        num_old_classes=num_old_classes,
        num_total_seen_classes=num_total_seen_classes,
        use_tts=use_tts,
        lr=current_lr
    )

    print(f"Client{client_id} training accuracy: {acc:.2f}%")
    return local_model

def train_task(global_model, task_train_dataset, task_test_datasets, task_id, device, replay_buffer, results_list, config_use_tts, method_name, dirichlet_alpha, num_old_classes=0, num_total_seen_classes=7):
    client_datasets = split_task_dataset_dirichlet(task_train_dataset, num_clients=NUM_CLIENTS, alpha=dirichlet_alpha)

    client_loaders = []
    for c_id in range(NUM_CLIENTS):
        if len(client_datasets[c_id]) > 0:
            client_loaders.append(get_dataloader(client_datasets[c_id], batch_size=64, shuffle=True, drop_last=True))
        else:
            client_loaders.append(None)

    for round_id in range(NUM_ROUNDS_PER_TASK):
        print(f"\n========== Task {task_id} | Round {round_id + 1} ==========")
        
        min_lr = 0.001
        max_lr = 0.01
        current_lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * round_id / NUM_ROUNDS_PER_TASK))
        local_models = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CLIENTS) as executor:
            futures = []
            for client_id in range(NUM_CLIENTS):
                futures.append(executor.submit(
                    train_single_client,
                    client_id, client_loaders[client_id], global_model, task_id, device,
                    replay_buffer, num_old_classes, num_total_seen_classes, config_use_tts, current_lr
                ))
            
            for future in concurrent.futures.as_completed(futures):
                local_models.append(future.result())

        global_model = fedavg(local_models).to(device)
        print("\nFedAvg aggregation finished.")
        
        for eval_task_id, test_loader in task_test_datasets.items():
            test_acc = evaluate(global_model, test_loader, device)
            print(f"Global test accuracy on Task{eval_task_id}: {test_acc:.2f}%")
            
            results_list.append({
                "Method": method_name,
                "Train_Task": task_id,
                "Round": round_id + 1,
                "Eval_Task": eval_task_id,
                "Accuracy": test_acc
            })

    return global_model

def run_experiment(config_buffer_type, config_use_tts, num_tasks, dirichlet_alpha):
    method_name = config_buffer_type
    if config_use_tts:
        method_name += "+tts"
    method_name += f"_tabular_t{num_tasks}_a{dirichlet_alpha}"
    
    
    raw_task_classes = cfg['task_classes_splits'][num_tasks]
    task_classes = {int(k): v for k, v in raw_task_classes.items()}

    task_cumulative_classes = get_task_cumulative_classes(task_classes)

    print_experiment_config(config_buffer_type, config_use_tts, method_name, num_tasks, dirichlet_alpha, task_classes)

    start_time = time.time() 
    all_results = []
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading Tabular Data...")
    train_dataset, test_dataset = load_tabular_data(target_col="label")
    
    input_dim = train_dataset.input_dim
    print(f"Feature input dimension: {input_dim}")

    train_task_datasets = get_task_datasets(train_dataset, task_classes)
    test_task_datasets = get_task_datasets(test_dataset, task_classes)

    test_loaders = {
        tid: get_dataloader(dataset, batch_size=64, shuffle=False)
        for tid, dataset in test_task_datasets.items()
    }

    total_classes = sum(len(v) for v in task_classes.values())
    global_model = get_tabular_model(input_dim=input_dim, num_classes=total_classes).to(device)

    replay_buffer = ReplayBuffer(
        samples_per_task=150, 
        num_clients=NUM_CLIENTS,
        candidate_pool_size=150, 
    )
    
    seen_test_loaders = {}
    
    for task_id in range(num_tasks):
        print(f"\n==================== START TASK {task_id} ====================")
        seen_test_loaders[task_id] = test_loaders[task_id]
        
        num_old_classes = 0 if task_id == 0 else task_cumulative_classes[task_id - 1]
        num_total_seen_classes = task_cumulative_classes[task_id]

        current_replay_buffer = None if task_id == 0 else replay_buffer

        global_model = train_task(
            global_model=global_model,
            task_train_dataset=train_task_datasets[task_id],
            task_test_datasets=seen_test_loaders,
            task_id=task_id, 
            device=device,
            replay_buffer=current_replay_buffer,
            results_list=all_results,
            config_use_tts=config_use_tts,
            method_name=method_name,
            dirichlet_alpha=dirichlet_alpha,
            num_old_classes=num_old_classes,
            num_total_seen_classes=num_total_seen_classes
        )

        print("\n--- Constructing Replay Buffer ---")
        if config_buffer_type != 'none':
            task_client_datasets = split_task_dataset_dirichlet(
                train_task_datasets[task_id], num_clients=NUM_CLIENTS, alpha=dirichlet_alpha
            )

            current_task_budget = len(task_classes[task_id]) * 100 
            replay_buffer.samples_per_task = current_task_budget

            replay_buffer.add_task_dataset(
                task_id=task_id, 
                client_datasets=task_client_datasets, 
                model=global_model, 
                device=device,
                strategy=config_buffer_type
            )
            total_buffer_size = 0
            for c in range(NUM_CLIENTS):
                c_data = replay_buffer.get_client_replay_data(c)
                if c_data is not None:
                    total_buffer_size += len(c_data)
                    
            print(f"Total Replay buffer size across all clients after Task{task_id}: {total_buffer_size}")  
        else:
            print("Skipping Replay Buffer (Finetune mode).")
        
    csv_filename = f"report_metrics_{method_name.replace('+', '_')}.csv"
    
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "Train_Task", "Round", "Eval_Task", "Accuracy"])
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\nAll metrics saved successfully to {csv_filename}")

    end_time = time.time()
    total_time_seconds = end_time - start_time
    hours = int(total_time_seconds // 3600)
    minutes = int((total_time_seconds % 3600) // 60)
    seconds = int(total_time_seconds % 60)
    
    print("=" * 50)
    print(f"EXPERIMENT COMPLETED FOR CONFIG {method_name}.")
    print(f"Total time elapsed: {hours}h {minutes}m {seconds}s")
    print("=" * 50)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True) 
    
    # Read iterative structures from YAML config
    task_configs = cfg.get('task_configs', [5])
    alpha_configs = cfg.get('alpha_configs', [0.5])
    configs_to_run = [(exp['buffer_type'], exp['use_tts']) for exp in cfg.get('experiments', [])]

    for num_tasks in task_configs:
        for alpha in alpha_configs:
            for buffer_type, use_tts in configs_to_run:
                print("\n\n" + "#" * 60)
                print(f"### LAUNCHING RUN: Tasks={num_tasks}, Alpha={alpha}, Buffer={buffer_type}, TTS={use_tts} ###")
                print("#" * 60)
                try:
                    run_experiment(
                        config_buffer_type=buffer_type, 
                        config_use_tts=use_tts, 
                        num_tasks=num_tasks, 
                        dirichlet_alpha=alpha
                    )
                except Exception as e:
                    print(f"Error captured in config T={num_tasks}, Alpha={alpha}, Buffer={buffer_type}+{use_tts}: {e}")