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

# Maps task_id to the total number of old classes accumulated before that task
TASK_CUMULATIVE_CLASSES = {
    0: 4,   # Total classes up to Task 0 (0,1,2,3)
    1: 7,   # Total classes up to Task 1 (0,1,2,3,4,5,6)
    2: 10   # Total classes up to Task 2 (0-9)
}

NUM_CLIENTS = 2
NUM_ROUNDS_PER_TASK = 2
DIRICHLET_ALPHA = 0.5 # Default hyperparameter for data heterogeneity


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
    # Use Non-IID Dirichlet distribution here instead of uniform split
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
            
            # Catch cases where a client might receive zero data for a task due to severe non-IID
            if len(client_datasets[client_id]) == 0:
                print(f"Skipping Client{client_id} due to 0 samples under Dirichlet Alpha={DIRICHLET_ALPHA}")
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
                num_total_seen_classes=num_total_seen_classes
            )

            print(f"Client{client_id} training accuracy: {acc:.2f}")

            local_models.append(local_model)

        global_model = fedavg(local_models).to(device)

        print("\nFedAvg aggregation finished.")
        print("Global model updated successfully.")

        for eval_task_id, test_loader in task_test_datasets.items():
            test_acc = evaluate(global_model, test_loader, device)
            print(f"Global model test accuracy on Task{eval_task_id}: {test_acc:.2f}")

    return global_model


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, test_dataset = load_cifar10()
    train_task_datasets = get_task_datasets(train_dataset, TASK_CLASSES)
    test_task_datasets = get_task_datasets(test_dataset, TASK_CLASSES)

    test_loaders = {}
    for task_id, dataset in test_task_datasets.items():
        test_loaders[task_id] = get_dataloader(dataset, batch_size=64, shuffle=False)

    global_model = get_model(num_classes=10).to(device)

    # exact GDR replay buffer
    replay_buffer = ReplayBuffer(
        samples_per_task=500,
        num_clients=NUM_CLIENTS,
        candidate_pool_size=300,
    )

    # -------- Train Task 0 --------
    print("\n==================== START TASK 0 ====================")
    seen_test_loaders = {0: test_loaders[0]}
    
    # Task 0: Total classes seen is 4 (classes 0, 1, 2, 3)
    global_model = train_task(
        global_model,
        train_task_datasets[0],
        seen_test_loaders,
        task_id=0,
        device=device,
        replay_dataset=None,
        num_old_classes=0,
        num_total_seen_classes=TASK_CUMULATIVE_CLASSES[0]
    )

    # Re-slice client datasets for Task 0 using Dirichlet for exact GDR use
    task0_client_datasets = split_task_dataset_dirichlet(
        train_task_datasets[0],
        num_clients=NUM_CLIENTS,
        alpha=DIRICHLET_ALPHA
    )

    replay_buffer.add_task_dataset(
        task_id=0,
        client_datasets=task0_client_datasets,
        model=global_model,
        device=device,
    )
    replay_dataset = replay_buffer.get_all_replay_data()

    print(f"\nReplay buffer size after Task0: {len(replay_dataset)}")
    replay_buffer.print_buffer_class_distribution(task_id=0)

    # -------- Train Task 1 --------
    print("\n==================== START TASK 1 ====================")
    seen_test_loaders = {0: test_loaders[0], 1: test_loaders[1]}
    
    global_model = train_task(
        global_model,
        train_task_datasets[1],
        seen_test_loaders,
        task_id=1,
        device=device,
        replay_dataset=replay_dataset,
        num_old_classes=TASK_CUMULATIVE_CLASSES[0],
        num_total_seen_classes=TASK_CUMULATIVE_CLASSES[1]
    )

    # Re-slice client datasets for Task 1 using Dirichlet for exact GDR use
    task1_client_datasets = split_task_dataset_dirichlet(
        train_task_datasets[1],
        num_clients=NUM_CLIENTS,
        alpha=DIRICHLET_ALPHA
    )

    replay_buffer.add_task_dataset(
        task_id=1,
        client_datasets=task1_client_datasets,
        model=global_model,
        device=device,
    )
    replay_dataset = replay_buffer.get_all_replay_data()

    print(f"\nReplay buffer size after Task1: {len(replay_dataset)}")
    replay_buffer.print_buffer_class_distribution(task_id=1)

    # -------- Train Task 2 --------
    print("\n==================== START TASK 2 ====================")
    seen_test_loaders = {0: test_loaders[0], 1: test_loaders[1], 2: test_loaders[2]}
    
    global_model = train_task(
        global_model,
        train_task_datasets[2],
        seen_test_loaders,
        task_id=2,
        device=device,
        replay_dataset=replay_dataset,
        num_old_classes=TASK_CUMULATIVE_CLASSES[1],
        num_total_seen_classes=TASK_CUMULATIVE_CLASSES[2]
    )
if __name__ == "__main__":
    main()