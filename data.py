from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np
from torch.utils.data import Subset

def load_cifar10():
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    return train_dataset, test_dataset


def get_task_datasets(dataset, task_classes):
    """
    dataset: CIFAR10 dataset
    task_classes: dict, e.g.
        {
            0: [0, 1, 2, 3],
            1: [4, 5, 6],
            2: [7, 8, 9],
        }
    return:
        dict of task_id -> Subset
    """
    task_datasets = {}

    targets = dataset.targets  # CIFAR10 labels list

    for task_id, classes in task_classes.items():
        indices = [i for i, label in enumerate(targets) if label in classes]
        task_datasets[task_id] = Subset(dataset, indices)

    return task_datasets


from torch.utils.data import random_split


def split_task_dataset_among_clients(task_dataset, num_clients=2):
    """
    把一个 task 的 dataset 平均分给多个 clients
    return: list of client subsets
    """
    total_size = len(task_dataset)
    base_size = total_size // num_clients
    sizes = [base_size] * num_clients

    # 把余数加到最后一个 client
    sizes[-1] += total_size - sum(sizes)

    client_subsets = random_split(task_dataset, sizes)

    return client_subsets


from torch.utils.data import DataLoader


def get_dataloader(dataset, batch_size=64, shuffle=True):
    import torch

    g = torch.Generator()
    g.manual_seed(42)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, generator=g
    )

def split_task_dataset_dirichlet(task_dataset, num_clients=2, alpha=0.5, seed=42):
    np.random.seed(seed)
    
    original_dataset = task_dataset.dataset
    indices = np.array(task_dataset.indices)
    
    labels = np.array([original_dataset.targets[i] for i in indices])
    classes = np.unique(labels)
    
    client_indices_map = {i: [] for i in range(num_clients)}
    
    for c in classes:
        idx_c = np.where(labels == c)[0]
        np.random.shuffle(idx_c)
        
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        num_samples_per_client = np.round(proportions * len(idx_c)).astype(int)
        
        diff = len(idx_c) - num_samples_per_client.sum()
        while diff > 0:
            client_id = np.random.randint(num_clients)
            num_samples_per_client[client_id] += 1
            diff -= 1
        while diff < 0:
            client_id = np.random.randint(num_clients)
            if num_samples_per_client[client_id] > 0:
                num_samples_per_client[client_id] -= 1
                diff += 1
                
        start_idx = 0
        for i in range(num_clients):
            end_idx = start_idx + num_samples_per_client[i]
            client_idcs = indices[idx_c[start_idx:end_idx]]
            client_indices_map[i].extend(client_idcs.tolist())
            start_idx = end_idx
            
    client_subsets = []
    for i in range(num_clients):
        np.random.shuffle(client_indices_map[i])
        client_subsets.append(Subset(original_dataset, client_indices_map[i]))
        
    return client_subsets