import random
from collections import Counter
from torch.utils.data import ConcatDataset, Subset
from typing import Union
import torch

from gdr import (
    extract_all_client_features,
    build_all_client_pseudo_features_exact,
    concat_client_pseudo_features,
    server_svd_global,
    split_global_U_to_clients,
    compute_client_leverage_scores,
    select_replay_per_client,
)


class ReplayBuffer:
    def __init__(
        self, samples_per_task=500, num_clients=2, candidate_pool_size=300
    ):  #  TODO    exact GDR 测试就是在 200 样本上成功的。先让 end-to-end 跑通，后面再慢慢把 200 提高到 300 / 500。
        self.samples_per_task = samples_per_task
        self.num_clients = num_clients
        self.candidate_pool_size = candidate_pool_size
        self.task_buffers = {}

    def add_task_dataset(self, task_id, client_datasets, model=None, device: Union[torch.device, str]="cpu", strategy="gdr"):
            if model is None and strategy == "gdr":
                raise ValueError("Exact GDR replay requires a model.")

            candidate_subsets = []
            for dataset in client_datasets:
                total_size = len(dataset)
                pool_size = min(self.candidate_pool_size, total_size)
                indices = random.sample(range(total_size), pool_size)
                candidate_subsets.append(Subset(dataset, indices))

            samples_per_client = self.samples_per_task // self.num_clients
            selected_subsets = []

            # baseline + GDR
            if strategy == "gdr":
                client_features, client_labels = extract_all_client_features(model, candidate_subsets, device)
                client_pseudo_features = build_all_client_pseudo_features_exact(client_features, target_dim=128)
                global_pseudo = concat_client_pseudo_features(client_pseudo_features)
                client_sizes = [pseudo.shape[0] for pseudo in client_pseudo_features]

                U_global = server_svd_global(global_pseudo, k=32)
                client_U_list = split_global_U_to_clients(U_global, client_sizes)

                for client_id in range(self.num_clients):
                    scores = compute_client_leverage_scores(client_U_list[client_id])
                    labels = client_labels[client_id]
                    selected_local_indices = select_replay_per_client(scores, labels, samples_per_client=samples_per_client)
                    selected_subsets.append(Subset(candidate_subsets[client_id], selected_local_indices))
            
            # baseline + TTS
            elif strategy == "random":
                for client_id in range(self.num_clients):
                    client_ds = candidate_subsets[client_id]
                    pool_size = len(client_ds)
                    actual_samples = min(samples_per_client, pool_size)
                    random_indices = random.sample(range(pool_size), actual_samples)
                    selected_subsets.append(Subset(client_ds, random_indices))

            self.task_buffers[task_id] = ConcatDataset(selected_subsets)

    def get_all_replay_data(self):
        if len(self.task_buffers) == 0:
            return None
        return ConcatDataset(list(self.task_buffers.values()))

    def print_buffer_class_distribution(self, task_id):
        if task_id not in self.task_buffers:
            print(f"No buffer found for Task {task_id}")
            return

        subset = self.task_buffers[task_id]
        counter = Counter()

        for i in range(len(subset)):
            _, label = subset[i]
            counter[label] += 1

        print(f"Class distribution in replay buffer for Task {task_id}:")
        for label, count in sorted(counter.items()):
            print(f"  Class {label}: {count}")
