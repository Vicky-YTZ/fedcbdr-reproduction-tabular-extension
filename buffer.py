import random
from collections import Counter
from torch.utils.data import ConcatDataset, Subset
from typing import Union
import torch
from model import TabularMLP, extract_tabular_features, extract_features

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
    def __init__(self, samples_per_task=2000, num_clients=5, candidate_pool_size=2000):
        self.samples_per_task = samples_per_task
        self.num_clients = num_clients
        self.candidate_pool_size = candidate_pool_size
        
        self.client_buffers = {i: [] for i in range(num_clients)}

    def add_task_dataset(self, task_id, client_datasets, model=None, device="cpu", strategy="gdr"):
        if model is None and strategy == "gdr":
            raise ValueError("Exact GDR replay requires a model.")

        # 1. Rút gọn dữ liệu thành Candidate Pool để tiết kiệm thời gian tính toán
        candidate_subsets = []
        for dataset in client_datasets:
            total_size = len(dataset)
            pool_size = min(self.candidate_pool_size, total_size)
            indices = random.sample(range(total_size), pool_size)
            candidate_subsets.append(Subset(dataset, indices))

        # Tính toán chỉ tiêu số lượng ảnh cho mỗi Client
        samples_per_client = self.samples_per_task // self.num_clients

        # -------------------------------------------------------------
        # CHIẾN LƯỢC 1: Global-perspective Data Replay (GDR) - Theo bài báo
        # -------------------------------------------------------------
        if strategy == "gdr":
            # Trích xuất đặc trưng và mã hóa bảo mật (Pseudo Features)
            if isinstance(model, TabularMLP):
                current_extract_fn = extract_tabular_features
            else:
                current_extract_fn = extract_features
            client_features, client_labels = extract_all_client_features(model, candidate_subsets, device, extract_fn=current_extract_fn)
            client_pseudo_features = build_all_client_pseudo_features_exact(client_features, target_dim=128)
            global_pseudo = concat_client_pseudo_features(client_pseudo_features)
            client_sizes = [pseudo.shape[0] for pseudo in client_pseudo_features]

            # Server chạy SVD và gửi ma trận U trả về cho Client
            U_global = server_svd_global(global_pseudo, k=32)
            client_U_list = split_global_U_to_clients(U_global, client_sizes)

            for client_id in range(self.num_clients):
                # Client tự tính điểm Leverage và bốc thăm ảnh
                scores = compute_client_leverage_scores(client_U_list[client_id])
                labels = client_labels[client_id]
                selected_local_indices = select_replay_per_client(scores, labels, samples_per_client=samples_per_client)
                
                # LƯU VÀO NGĂN TỦ RIÊNG CỦA ĐÚNG CLIENT ĐÓ
                subset = Subset(candidate_subsets[client_id], selected_local_indices)
                self.client_buffers[client_id].append(subset)
        
        # -------------------------------------------------------------
        # CHIẾN LƯỢC 2: Random Selection (Baseline)
        # -------------------------------------------------------------
        elif strategy == "random":
            import time
            random.seed(int(time.time() * 1000) % 10000)
            for client_id in range(self.num_clients):
                client_ds = candidate_subsets[client_id]
                pool_size = len(client_ds)
                actual_samples = min(samples_per_client, pool_size)
                
                random_indices = random.sample(range(pool_size), actual_samples)
                
                # LƯU VÀO NGĂN TỦ RIÊNG CỦA ĐÚNG CLIENT ĐÓ
                subset = Subset(client_ds, random_indices)
                self.client_buffers[client_id].append(subset)



    def get_client_replay_data(self, client_id):
        """
        Trả về dữ liệu bộ nhớ ĐỘC LẬP của riêng Client đó để phục vụ huấn luyện (Train)
        """
        if len(self.client_buffers[client_id]) == 0:
            return None
        return ConcatDataset(self.client_buffers[client_id])

    def print_buffer_class_distribution(self, client_id):
        """
        Hàm Debug: Kiểm tra xem ngăn tủ của Client có thực sự phân bổ đều (Class-Balanced) không
        """
        client_data = self.get_client_replay_data(client_id)
        if client_data is None:
            print(f"No buffer found for Client {client_id}")
            return

        counter = Counter()

        for i in range(len(client_data)):
            _, label = client_data[i]
            counter[label] += 1

        print(f"\nClass distribution in replay buffer for Client {client_id}:")
        for label, count in sorted(counter.items()):
            print(f"  Class {label}: {count}")
