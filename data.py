from torchvision import datasets, transforms
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
