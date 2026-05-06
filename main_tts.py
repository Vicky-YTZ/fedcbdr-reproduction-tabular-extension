import copy
import random
import numpy as np
import torch

from model import get_model
from data import (
    load_cifar10,
    get_task_datasets,
    split_task_dataset_dirichlet,
    get_dataloader,
)
from train import train_one_epoch_tts
from federated import fedavg
from eval import evaluate
from buffer import ReplayBuffer

TASK_CLASSES = {0: [0, 1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9]}
# 'none' (Finetune = none + false), 'random' (+TTS), 'gdr' (+GDR / +GDR+TTS)
CONFIG_BUFFER_TYPE = 'gdr'
CONFIG_USE_TTS = True

def get_task_cumulative_classes(task_classes):
    cumulative = {}
    total = 0
    for task_id in sorted(task_classes.keys()):
        total += len(task_classes[task_id])
        cumulative[task_id] = total
    return cumulative

TASK_CUMULATIVE_CLASSES = get_task_cumulative_classes(TASK_CLASSES)

NUM_CLIENTS = 2
NUM_ROUNDS_PER_TASK = 2
DIRICHLET_ALPHA = 0.5 

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_task(
    global_model,
    task_train_dataset,
    task_test_datasets,
    task_id,
    device,
    replay_dataset=None,
    num_old_classes=0,
    num_total_seen_classes=10
):
    client_datasets = split_task_dataset_dirichlet(
        task_train_dataset, 
        num_clients=NUM_CLIENTS, 
        alpha=DIRICHLET_ALPHA
    )

    for round_id in range(NUM_ROUNDS_PER_TASK):
        print(f"\n========== Task {task_id} | Round {round_id + 1} ==========")

        local_models = []

        for client_id in range(NUM_CLIENTS):
            print(f"\nStart training Client{client_id} Task{task_id}")
            
            if len(client_datasets[client_id]) == 0:
                print(f"Skipping Client{client_id} due to 0 samples.")
                local_models.append(copy.deepcopy(global_model).to(device))
                continue

            client_loader = get_dataloader(
                client_datasets[client_id], batch_size=64, shuffle=True
            )

            local_model = copy.deepcopy(global_model).to(device)

            acc = train_one_epoch_tts(
                local_model,
                client_loader,
                device,
                replay_dataset=replay_dataset,
                batch_size=64,
                task_id=task_id,
                num_old_classes=num_old_classes,
                num_total_seen_classes=num_total_seen_classes,
                use_tts=CONFIG_USE_TTS
            )

            print(f"Client{client_id} training accuracy: {acc:.2f}")
            local_models.append(local_model)

        global_model = fedavg(local_models).to(device)

        print("\nFedAvg aggregation finished.")
        
        for eval_task_id, test_loader in task_test_datasets.items():
            test_acc = evaluate(global_model, test_loader, device)
            print(f"Global test accuracy on Task{eval_task_id}: {test_acc:.2f}")

    return global_model


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, test_dataset = load_cifar10()
    train_task_datasets = get_task_datasets(train_dataset, TASK_CLASSES)
    test_task_datasets = get_task_datasets(test_dataset, TASK_CLASSES)

    test_loaders = {
        tid: get_dataloader(dataset, batch_size=64, shuffle=False)
        for tid, dataset in test_task_datasets.items()
    }

    # Model Initialization
    global_model = get_model(num_classes=10).to(device)

    # exact GDR replay buffer
    replay_buffer = ReplayBuffer(
        samples_per_task=500,
        num_clients=NUM_CLIENTS,
        candidate_pool_size=300,
    )
    
    replay_dataset = None
    seen_test_loaders = {}

    num_tasks = len(TASK_CLASSES)
    
    for task_id in range(num_tasks):
        print(f"\n==================== START TASK {task_id} ====================")
        
        seen_test_loaders[task_id] = test_loaders[task_id]
        
        num_old_classes = 0 if task_id == 0 else TASK_CUMULATIVE_CLASSES[task_id - 1]
        num_total_seen_classes = TASK_CUMULATIVE_CLASSES[task_id]

        # Training
        global_model = train_task(
            global_model=global_model,
            task_train_dataset=train_task_datasets[task_id],
            task_test_datasets=seen_test_loaders,
            task_id=task_id,
            device=device,
            replay_dataset=replay_dataset,
            num_old_classes=num_old_classes,
            num_total_seen_classes=num_total_seen_classes
        )

        print("\n--- Constructing Replay Buffer ---")
        if CONFIG_BUFFER_TYPE != 'none':
            task_client_datasets = split_task_dataset_dirichlet(
                train_task_datasets[task_id],
                num_clients=NUM_CLIENTS,
                alpha=DIRICHLET_ALPHA
            )

            replay_buffer.add_task_dataset(
                task_id=task_id,
                client_datasets=task_client_datasets,
                model=global_model,
                device=device,
                strategy=CONFIG_BUFFER_TYPE
            )
            
            replay_dataset = replay_buffer.get_all_replay_data()
            print(f"Replay buffer size after Task{task_id}: {len(replay_dataset)}")
        else:
            replay_dataset = None
            print("Skipping Replay Buffer (Finetune mode).")


if __name__ == "__main__":
    main()