import copy
import random
import numpy as np
import torch
import math
import concurrent.futures
import time
import csv
import yaml

from model import get_model, expand_model_classifier
from data import (
    load_cifar10,
    load_cifar100,
    load_tinyimagenet,
    get_task_datasets,
    split_task_dataset_dirichlet,
    get_dataloader,
)
from train import train_one_epoch_tts
from federated import fedavg
from eval import evaluate
from buffer import ReplayBuffer

# ---------------- CONFIG ---------------- 
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)['vision']

DATASET = cfg['dataset']
NUM_CLIENTS = cfg['num_clients']
NUM_ROUNDS_PER_TASK = cfg['num_rounds_per_task']
DIRICHLET_ALPHA = cfg['dirichlet_alpha']

if DATASET == "cifar10":
    TASK_CLASSES = cfg['task_classes']['cifar10']
elif DATASET == "cifar100":
    num_tasks = cfg['task_classes']['cifar100_split']
    c_per_task = 100 // num_tasks
    TASK_CLASSES = {i: list(range(i * c_per_task, (i + 1) * c_per_task)) for i in range(num_tasks)}
elif DATASET == "tinyimagenet":
    num_tasks = cfg['task_classes']['tinyimagenet_split']
    c_per_task = 200 // num_tasks
    TASK_CLASSES = {i: list(range(i * c_per_task, (i + 1) * c_per_task)) for i in range(num_tasks)}

# Ensure dictionary keys are integers (YAML parses them as strings sometimes)
TASK_CLASSES = {int(k): v for k, v in TASK_CLASSES.items()}

def get_task_cumulative_classes(task_classes):
    cumulative = {}
    total = 0
    for task_id in sorted(task_classes.keys()):
        total += len(task_classes[task_id])
        cumulative[task_id] = total
    return cumulative

TASK_CUMULATIVE_CLASSES = get_task_cumulative_classes(TASK_CLASSES)

def print_experiment_config(buffer_type, use_tts, method_name):
    print("=" * 50)
    print(" " * 10 + "EXPERIMENT CONFIGURATIONS")
    print("=" * 50)
    print(f"Dataset                              : {DATASET} (Vision)")
    print(f"Method Name                          : {method_name}")
    print(f"Buffer Strategy                      : {buffer_type}")
    print(f"Use TTS                              : {use_tts}")
    print(f"Number of Clients (NUM_CLIENTS)      : {NUM_CLIENTS}")
    print(f"Rounds per Task (NUM_ROUNDS_PER_TASK): {NUM_ROUNDS_PER_TASK}")
    print(f"Dirichlet Alpha (DIRICHLET_ALPHA)    : {DIRICHLET_ALPHA}")
    print(f"Task Classes                         : {TASK_CLASSES}")
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
    if client_loader is None or len(client_loader.dataset) == 0:
        return copy.deepcopy(global_model).to(device)

    local_model = copy.deepcopy(global_model).to(device)
    client_replay = replay_buffer.get_client_replay_data(client_id) if replay_buffer else None

    for _ in range(1):
        train_one_epoch_tts(
            local_model, client_loader, device, replay_dataset=client_replay,
            batch_size=64, task_id=task_id, num_old_classes=num_old_classes,
            num_total_seen_classes=num_total_seen_classes, use_tts=use_tts, lr=current_lr
        )
    return local_model

def train_task(global_model, task_train_dataset, task_test_datasets, task_id, device, replay_buffer, results_list, config_use_tts, method_name, num_old_classes=0, num_total_seen_classes=10):
    client_datasets = split_task_dataset_dirichlet(task_train_dataset, num_clients=NUM_CLIENTS, alpha=DIRICHLET_ALPHA)

    client_loaders = [get_dataloader(c, batch_size=64, shuffle=True) if len(c) > 0 else None for c in client_datasets]

    for round_id in range(NUM_ROUNDS_PER_TASK):
        print(f"\n========== Task {task_id} | Round {round_id + 1} ==========")
        current_lr = 0.001 + 0.5 * (0.01 - 0.001) * (1 + math.cos(math.pi * round_id / NUM_ROUNDS_PER_TASK))
        local_models = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CLIENTS) as executor:
            futures = [executor.submit(
                train_single_client, i, client_loaders[i], global_model, task_id, device,
                replay_buffer, num_old_classes, num_total_seen_classes, config_use_tts, current_lr
            ) for i in range(NUM_CLIENTS)]

            for future in concurrent.futures.as_completed(futures):
                local_models.append(future.result())

        global_model = fedavg(local_models).to(device)
        
        for eval_task_id, test_loader in task_test_datasets.items():
            test_acc = evaluate(global_model, test_loader, device, num_old_classes, config_use_tts, 0.5, 1.5)
            print(f"Global test accuracy on Task{eval_task_id}: {test_acc:.2f}")
            results_list.append({"Method": method_name, "Train_Task": task_id, "Round": round_id + 1, "Eval_Task": eval_task_id, "Accuracy": test_acc})

    return global_model

def run_experiment(buffer_type, use_tts):
    method_name = f"{buffer_type}{'+tts' if use_tts else ''}"
    print_experiment_config(buffer_type, use_tts, method_name)

    start_time = time.time()
    all_results = []
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if DATASET == "cifar10": train_dataset, test_dataset = load_cifar10()
    elif DATASET == "cifar100": train_dataset, test_dataset = load_cifar100()
    elif DATASET == "tinyimagenet": train_dataset, test_dataset = load_tinyimagenet()
        
    train_tasks = get_task_datasets(train_dataset, TASK_CLASSES)
    test_tasks = get_task_datasets(test_dataset, TASK_CLASSES)
    test_loaders = {k: get_dataloader(v, batch_size=64, shuffle=False) for k, v in test_tasks.items()}

    global_model = get_model(num_classes=len(TASK_CLASSES[0])).to(device)
    replay_buffer = ReplayBuffer(samples_per_task=2000, num_clients=NUM_CLIENTS, candidate_pool_size=2000)

    seen_test_loaders = {}
    
    for task_id in range(len(TASK_CLASSES)):
        print(f"\n==================== START TASK {task_id} ====================")
        seen_test_loaders[task_id] = test_loaders[task_id]

        num_old_classes = 0 if task_id == 0 else TASK_CUMULATIVE_CLASSES[task_id - 1]
        num_total_seen_classes = TASK_CUMULATIVE_CLASSES[task_id]
        
        global_model = expand_model_classifier(global_model, num_total_seen_classes, device)

        global_model = train_task(
            global_model, train_tasks[task_id], seen_test_loaders, task_id, device,
            replay_buffer if task_id > 0 else None, all_results, use_tts, method_name,
            num_old_classes, num_total_seen_classes
        )

        if buffer_type != "none":
            client_ds = split_task_dataset_dirichlet(train_tasks[task_id], NUM_CLIENTS, DIRICHLET_ALPHA)
            replay_buffer.samples_per_task = len(TASK_CLASSES[task_id]) * 50
            replay_buffer.add_task_dataset(task_id, client_ds, global_model, device, buffer_type)

    csv_name = f"report_metrics_vision_{method_name.replace('+', '_')}.csv"
    with open(csv_name, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "Train_Task", "Round", "Eval_Task", "Accuracy"])
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\nAll metrics saved to {csv_name}")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    
    for exp in cfg['experiments']:
        try:
            run_experiment(exp['buffer_type'], exp['use_tts'])
        except Exception as e:
            print(f"Failed to run vision experiment {exp}: {e}")