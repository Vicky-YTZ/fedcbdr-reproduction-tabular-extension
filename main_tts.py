import copy
import random
import numpy as np
import torch
import math
import concurrent.futures
import time
import csv

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

DATASET = "tinyimagenet"
if DATASET == "cifar10":
    num_tasks = 3
    TASK_CLASSES = {0: [0, 1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9]}
elif DATASET == "cifar100":
    num_tasks = 5
    classes_per_task = 100 // num_tasks
    TASK_CLASSES = {
        i: list(range(i * classes_per_task, (i + 1) * classes_per_task))
        for i in range(num_tasks)
    }
elif DATASET == "tinyimagenet":
    num_tasks = 10
    classes_per_task = 200 // num_tasks
    TASK_CLASSES = {
        i: list(range(i * classes_per_task, (i + 1) * classes_per_task))
        for i in range(num_tasks)
    }

# Buffer strategy:
# 'none'       = finetune baseline
# 'random'     = random replay baseline
# 'gdr'        = original GDR with leverage-score selection
# 'gdr_kmeans' = extension: GDR(SVD) + k-means selection
# 'kmeans'     = local feature k-means selection
# 'svd_kmeans' = local SVD/PCA + k-means selection
CONFIG_BUFFER_TYPE = "none"
CONFIG_USE_TTS = False

METHOD_NAME = CONFIG_BUFFER_TYPE
if CONFIG_USE_TTS:
    METHOD_NAME += "+tts"


def print_experiment_config():
    print("=" * 50)
    print(" " * 10 + "EXPERIMENT CONFIGURATIONS")
    print("=" * 50)
    print(f"Buffer Strategy (CONFIG_BUFFER_TYPE) : {CONFIG_BUFFER_TYPE}")
    print(f"Number of Tasks                      : {num_tasks}")
    print(f"Use TTS (CONFIG_USE_TTS)             : {CONFIG_USE_TTS}")
    print(f"Number of Clients (NUM_CLIENTS)      : {NUM_CLIENTS}")
    print(f"Rounds per Task (NUM_ROUNDS_PER_TASK): {NUM_ROUNDS_PER_TASK}")
    print(f"Dirichlet Alpha (DIRICHLET_ALPHA)    : {DIRICHLET_ALPHA}")
    print(f"Task Classes (TASK_CLASSES)          : {TASK_CLASSES}")
    print(f"Seed                                 : 42")
    print(f"Batch Size (hardcoded)               : 64")
    print(f"Local Epochs (hardcoded)             : 2")
    print("=" * 50 + "\n")


def get_task_cumulative_classes(task_classes):
    cumulative = {}
    total = 0
    for task_id in sorted(task_classes.keys()):
        total += len(task_classes[task_id])
        cumulative[task_id] = total
    return cumulative


TASK_CUMULATIVE_CLASSES = get_task_cumulative_classes(TASK_CLASSES)

NUM_CLIENTS = 5
NUM_ROUNDS_PER_TASK = 20
DIRICHLET_ALPHA = 1


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_single_client(
    client_id,
    client_loader,
    global_model,
    task_id,
    device,
    replay_buffer,
    num_old_classes,
    num_total_seen_classes,
    use_tts,
    current_lr,
):
    print(f"\nStart training Client{client_id} Task{task_id}")

    if client_loader is None or len(client_loader.dataset) == 0:
        print(f"Skipping Client{client_id} due to 0 samples.")
        return copy.deepcopy(global_model).to(device)

    local_model = copy.deepcopy(global_model).to(device)

    if replay_buffer is not None:
        client_replay = replay_buffer.get_client_replay_data(client_id)
    else:
        client_replay = None

    for _ in range(1):
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
            lr=current_lr,
        )

    print(f"Client{client_id} training accuracy: {acc:.2f}")
    return local_model


def train_task(
    global_model,
    task_train_dataset,
    task_test_datasets,
    task_id,
    device,
    replay_buffer,
    results_list,
    num_old_classes=0,
    num_total_seen_classes=10,
):
    client_datasets = split_task_dataset_dirichlet(
        task_train_dataset, num_clients=NUM_CLIENTS, alpha=DIRICHLET_ALPHA
    )

    client_loaders = []
    for c_id in range(NUM_CLIENTS):
        if len(client_datasets[c_id]) > 0:
            c_loader = get_dataloader(
                client_datasets[c_id], batch_size=64, shuffle=True
            )
            client_loaders.append(c_loader)
        else:
            client_loaders.append(None)

    for round_id in range(NUM_ROUNDS_PER_TASK):
        print(f"\n========== Task {task_id} | Round {round_id + 1} ==========")

        # --- Cosine Annealing Learning Rate ---
        min_lr = 0.001
        max_lr = 0.01
        current_lr = min_lr + 0.5 * (max_lr - min_lr) * (
            1 + math.cos(math.pi * round_id / NUM_ROUNDS_PER_TASK)
        )
        local_models = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CLIENTS) as executor:
            futures = []
            for client_id in range(NUM_CLIENTS):
                futures.append(
                    executor.submit(
                        train_single_client,
                        client_id=client_id,
                        client_loader=client_loaders[client_id],
                        global_model=global_model,
                        task_id=task_id,
                        device=device,
                        replay_buffer=replay_buffer,
                        num_old_classes=num_old_classes,
                        num_total_seen_classes=num_total_seen_classes,
                        use_tts=CONFIG_USE_TTS,
                        current_lr=current_lr,
                    )
                )

            # Collect local models as they finish training
            for future in concurrent.futures.as_completed(futures):
                local_models.append(future.result())

        # --- FedAvg ---
        global_model = fedavg(local_models).to(device)
        print("\nFedAvg aggregation finished.")

        # --- Global Evaluation ---
        for eval_task_id, test_loader in task_test_datasets.items():
            test_acc = evaluate(
                model=global_model,
                dataloader=test_loader,
                device=device,
                num_old_classes=num_old_classes,
                use_tts=CONFIG_USE_TTS,
                tau_old=0.5,
                tau_new=1.5,
            )
            print(f"Global test accuracy on Task{eval_task_id}: {test_acc:.2f}")

            results_list.append(
                {
                    "Method": METHOD_NAME,
                    "Train_Task": task_id,
                    "Round": round_id + 1,
                    "Eval_Task": eval_task_id,
                    "Accuracy": test_acc,
                }
            )

    return global_model


def main():
    start_time = time.time()
    all_results = []
    print_experiment_config()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if DATASET == "cifar10":
        train_dataset, test_dataset = load_cifar10()
    elif DATASET == "cifar100":
        train_dataset, test_dataset = load_cifar100()
    elif DATASET == "tinyimagenet":
        train_dataset, test_dataset = load_tinyimagenet()
    train_task_datasets = get_task_datasets(train_dataset, TASK_CLASSES)
    test_task_datasets = get_task_datasets(test_dataset, TASK_CLASSES)

    test_loaders = {
        tid: get_dataloader(dataset, batch_size=64, shuffle=False)
        for tid, dataset in test_task_datasets.items()
    }

    # Initialize the model with the number of classes for Task 0
    num_initial_classes = len(TASK_CLASSES[0])
    global_model = get_model(num_classes=num_initial_classes).to(device)

    # Replay buffer
    replay_buffer = ReplayBuffer(
        samples_per_task=2000,
        num_clients=NUM_CLIENTS,
        candidate_pool_size=2000,
    )

    seen_test_loaders = {}
    num_tasks = len(TASK_CLASSES)

    for task_id in range(num_tasks):
        print(f"\n==================== START TASK {task_id} ====================")

        seen_test_loaders[task_id] = test_loaders[task_id]

        num_old_classes = 0 if task_id == 0 else TASK_CUMULATIVE_CLASSES[task_id - 1]
        num_total_seen_classes = TASK_CUMULATIVE_CLASSES[task_id]

        # Expand model classifier for new classes
        global_model = expand_model_classifier(
            global_model, num_total_seen_classes, device
        )

        # Training
        global_model = train_task(
            global_model=global_model,
            task_train_dataset=train_task_datasets[task_id],
            task_test_datasets=seen_test_loaders,
            task_id=task_id,
            device=device,
            replay_buffer=replay_buffer if task_id > 0 else None,
            results_list=all_results,
            num_old_classes=num_old_classes,
            num_total_seen_classes=num_total_seen_classes,
        )

        print("\n--- Constructing Replay Buffer ---")
        if CONFIG_BUFFER_TYPE != "none":
            task_client_datasets = split_task_dataset_dirichlet(
                train_task_datasets[task_id],
                num_clients=NUM_CLIENTS,
                alpha=DIRICHLET_ALPHA,
            )

            current_task_budget = len(TASK_CLASSES[task_id]) * 50
            replay_buffer.samples_per_task = current_task_budget

            replay_buffer.add_task_dataset(
                task_id=task_id,
                client_datasets=task_client_datasets,
                model=global_model,
                device=device,
                strategy=CONFIG_BUFFER_TYPE,
            )

            # Calculate the total number of images stored across all clients
            total_buffer_size = 0
            for c in range(NUM_CLIENTS):
                c_data = replay_buffer.get_client_replay_data(c)
                if c_data is not None:
                    total_buffer_size += len(c_data)

            print(
                f"Total Replay buffer size across all clients after Task{task_id}: {total_buffer_size}"
            )
        else:
            print("Skipping Replay Buffer (Finetune mode).")

    # Save results to CSV file
    csv_filename = f"report_metrics_{METHOD_NAME.replace('+', '_')}.csv"

    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Method", "Train_Task", "Round", "Eval_Task", "Accuracy"]
        )
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nAll metrics saved successfully to {csv_filename}")

    end_time = time.time()
    total_time_seconds = end_time - start_time
    hours = int(total_time_seconds // 3600)
    minutes = int((total_time_seconds % 3600) // 60)
    seconds = int(total_time_seconds % 60)

    print("=" * 50)
    print(f"EXPERIMENT COMPLETED.")
    print(f"Total time elapsed: {hours}h {minutes}m {seconds}s")
    print("=" * 50)
    print_experiment_config()


if __name__ == "__main__":
    import torch.multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
