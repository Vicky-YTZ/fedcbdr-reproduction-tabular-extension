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

    def add_task_dataset(
        self, task_id, client_datasets, model=None, device="cpu", strategy="gdr"
    ):
        if model is None and strategy in [
            "gdr",
            "gdr_kmeans",
            "herding",
            "kmeans",
            "svd_kmeans",
        ]:
            raise ValueError(f"{strategy} replay requires a model.")

        # 1. Reduce data to Candidate Pool to save computation time
        candidate_subsets = []
        for dataset in client_datasets:
            total_size = len(dataset)
            pool_size = min(self.candidate_pool_size, total_size)
            indices = random.sample(range(total_size), pool_size)
            candidate_subsets.append(Subset(dataset, indices))

        # Compute the number of samples per Client
        samples_per_client = self.samples_per_task // self.num_clients

        # -------------------------------------------------------------
        # STRATEGY 1: Global-perspective Data Replay (GDR) - According to the paper
        # -------------------------------------------------------------
        if strategy == "gdr":
            # Extract features and secure encoding (Pseudo Features)
            if isinstance(model, TabularMLP):
                current_extract_fn = extract_tabular_features
            else:
                current_extract_fn = extract_features
            client_features, client_labels = extract_all_client_features(
                model, candidate_subsets, device, extract_fn=current_extract_fn
            )
            client_pseudo_features = build_all_client_pseudo_features_exact(
                client_features, target_dim=128
            )
            global_pseudo = concat_client_pseudo_features(client_pseudo_features)
            client_sizes = [pseudo.shape[0] for pseudo in client_pseudo_features]

            # Server runs SVD and returns matrix U to Client
            U_global = server_svd_global(global_pseudo, k=32)
            client_U_list = split_global_U_to_clients(U_global, client_sizes)

            for client_id in range(self.num_clients):
                # Client computes Leverage scores and samples data
                scores = compute_client_leverage_scores(client_U_list[client_id])
                labels = client_labels[client_id]
                selected_local_indices = select_replay_per_client(
                    scores, labels, samples_per_client=samples_per_client
                )

                # SAVE TO THE SPECIFIC CLIENT'S BUFFER
                subset = Subset(candidate_subsets[client_id], selected_local_indices)
                self.client_buffers[client_id].append(subset)
                # -------------------------------------------------------------
        # STRATEGY 1B: GDR(SVD) + k-means
        # Same GDR pseudo-feature + global SVD pipeline, but replace
        # leverage-score sampling with class-balanced k-means selection
        # on the global SVD latent representation U_k.
        # -------------------------------------------------------------
        elif strategy == "gdr_kmeans":
            if isinstance(model, TabularMLP):
                current_extract_fn = extract_tabular_features
            else:
                current_extract_fn = extract_features

            client_features, client_labels = extract_all_client_features(
                model,
                candidate_subsets,
                device,
                extract_fn=current_extract_fn,
            )

            client_pseudo_features = build_all_client_pseudo_features_exact(
                client_features,
                target_dim=128,
            )

            global_pseudo = concat_client_pseudo_features(client_pseudo_features)
            client_sizes = [pseudo.shape[0] for pseudo in client_pseudo_features]

            # This is the same global SVD step as GDR.
            U_global = server_svd_global(global_pseudo, k=32)
            client_U_list = split_global_U_to_clients(U_global, client_sizes)

            for client_id in range(self.num_clients):
                labels = client_labels[client_id]

                # Extension: use k-means on the GDR/SVD latent space instead of leverage scores.
                selected_local_indices = self.select_kmeans_per_client(
                    features=client_U_list[client_id],
                    labels=labels,
                    samples_per_client=samples_per_client,
                )

                if len(selected_local_indices) > 0:
                    subset = Subset(
                        candidate_subsets[client_id], selected_local_indices
                    )
                    self.client_buffers[client_id].append(subset)
        # -------------------------------------------------------------
        # STRATEGY 2: Random Selection (Baseline)
        # -------------------------------------------------------------
        elif strategy == "random":
            import time

            random.seed(int(time.time() * 1000) % 10000)
            for client_id in range(self.num_clients):
                client_ds = candidate_subsets[client_id]
                pool_size = len(client_ds)
                actual_samples = min(samples_per_client, pool_size)

                # Random selection
                random_indices = random.sample(range(pool_size), actual_samples)

                # SAVE TO THE SPECIFIC CLIENT'S BUFFER
                subset = Subset(client_ds, random_indices)
                self.client_buffers[client_id].append(subset)
        elif strategy == "herding":
            # 1. Enable model and extract features exactly like GDR
            if isinstance(model, TabularMLP):
                current_extract_fn = extract_tabular_features
            else:
                current_extract_fn = extract_features

            # This function is already available in gdr.py
            client_features, client_labels = extract_all_client_features(
                model, candidate_subsets, device, extract_fn=current_extract_fn
            )

            for client_id in range(self.num_clients):
                feats = client_features[client_id]
                lbls = client_labels[client_id]

                # 2. Apply Herding algorithm for selection (Separately for each Client)
                selected_local_indices = self.select_herding_per_client(
                    features=feats, labels=lbls, samples_per_client=samples_per_client
                )

                # 3. Save to the client's storage
                if len(selected_local_indices) > 0:
                    subset = Subset(
                        candidate_subsets[client_id], selected_local_indices
                    )
                    self.client_buffers[client_id].append(subset)

        elif strategy == "kmeans":
            if isinstance(model, TabularMLP):
                current_extract_fn = extract_tabular_features
            else:
                current_extract_fn = extract_features

            client_features, client_labels = extract_all_client_features(
                model, candidate_subsets, device, extract_fn=current_extract_fn
            )
            for client_id in range(self.num_clients):
                feats = client_features[client_id]
                lbls = client_labels[client_id]
                selected_local_indices = self.select_kmeans_per_client(
                    feats, lbls, samples_per_client
                )
                if len(selected_local_indices) > 0:
                    subset = Subset(
                        candidate_subsets[client_id], selected_local_indices
                    )
                    self.client_buffers[client_id].append(subset)
        elif strategy == "svd_kmeans":
            if isinstance(model, TabularMLP):
                current_extract_fn = extract_tabular_features
            else:
                current_extract_fn = extract_features

            client_features, client_labels = extract_all_client_features(
                model, candidate_subsets, device, extract_fn=current_extract_fn
            )
            for client_id in range(self.num_clients):
                feats = client_features[client_id]
                lbls = client_labels[client_id]
                selected_local_indices = self.select_svd_kmeans_per_client(
                    feats, lbls, samples_per_client
                )
                if len(selected_local_indices) > 0:
                    subset = Subset(
                        candidate_subsets[client_id], selected_local_indices
                    )
                    self.client_buffers[client_id].append(subset)

    def get_client_replay_data(self, client_id):
        """
        Return INDEPENDENT replay memory data of the specific Client for training
        """
        if len(self.client_buffers[client_id]) == 0:
            return None
        return ConcatDataset(self.client_buffers[client_id])

    def print_buffer_class_distribution(self, client_id):
        """
        Debug function: Check if the client's buffer is truly class-balanced
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

    def select_herding_per_client(self, features, labels, samples_per_client):
        """
        Herding Algorithm: Select a subset such that the mean vector
        of the selected elements is closest to the mean vector of the entire class.

        Args:
            features: Tensor of shape [N, d] (Note: not normalized yet)
            labels: Tensor of shape [N]
            samples_per_client: int (Total quota)
        Returns:
            selected_indices: List of sample positions (len <= samples_per_client)
        """
        from collections import defaultdict
        import numpy as np

        class_to_indices = defaultdict(list)
        for idx, lbl in enumerate(labels.tolist()):
            class_to_indices[int(lbl)].append(idx)

        classes = list(class_to_indices.keys())
        if len(classes) == 0 or samples_per_client == 0:
            return []

        # Evenly divide memory quota for classes
        base_quota = samples_per_client // len(classes)
        remainder = samples_per_client % len(classes)

        class_quotas = {cls: base_quota for cls in classes}
        if remainder > 0:
            lucky_cls = np.random.choice(classes, remainder, replace=False)
            for c in lucky_cls:
                class_quotas[int(c)] += 1

        selected_indices = []

        for cls in classes:
            quota = class_quotas[cls]
            indices_of_class = class_to_indices[cls]

            if len(indices_of_class) <= quota:
                # If less than quota, take all
                selected_indices.extend(indices_of_class)
                continue

            # Extract features specifically for this class
            class_features = features[indices_of_class]  # shape [N_c, d]

            # NORMALIZE DATA TO AVOID VECTOR DEVIATION IN SPACE
            class_features = torch.nn.functional.normalize(class_features, p=2, dim=1)

            # Calculate IDEAL CENTER (Mean Feature) of Original Samples (Class Center)
            class_mean = torch.mean(class_features, dim=0)  # [d]
            class_mean = torch.nn.functional.normalize(class_mean, p=2, dim=0)

            # Start looping to pick each sample sequentially using Herding algorithm
            selected_feat_sum = torch.zeros_like(
                class_mean
            )  # initialize sum vector = 0

            current_selected = []
            available_indices = list(range(len(indices_of_class)))

            for k in range(quota):
                best_idx = -1
                best_dist = float("inf")

                for idx_in_list in available_indices:
                    feat_i = class_features[idx_in_list]

                    # STANDARD iCaRL EQUATION:
                    # Calculate hypothetical mean vector if adding feat_i to the selected set
                    temp_mean = (selected_feat_sum + feat_i) / (k + 1)

                    # Goal: Distance from 'Hypothetical Mean' to 'Original Mean' must be MINIMAL
                    dist = torch.norm(class_mean - temp_mean, p=2).item()

                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx_in_list

                # Save the optimal sample
                current_selected.append(indices_of_class[best_idx])
                available_indices.remove(best_idx)

                # Update moving center (Most important so subsequent points spread out)
                selected_feat_sum += class_features[best_idx]

            selected_indices.extend(current_selected)

        # Fill remaining quota if class-balanced k-means selected too few samples.
        # This is important under non-IID splits, where some classes may have too few candidates.
        if len(selected_indices) < samples_per_client:
            selected_set = set(selected_indices)
            for idx in range(len(labels)):
                if len(selected_indices) >= samples_per_client:
                    break
                if idx not in selected_set:
                    selected_indices.append(idx)
                    selected_set.add(idx)

        return selected_indices[:samples_per_client]

    def select_kmeans_per_client(self, features, labels, samples_per_client):
        from collections import defaultdict
        import numpy as np
        from sklearn.cluster import KMeans

        class_to_indices = defaultdict(list)
        for idx, lbl in enumerate(labels.tolist()):
            class_to_indices[int(lbl)].append(idx)

        classes = list(class_to_indices.keys())
        if len(classes) == 0 or samples_per_client == 0:
            return []

        base_quota = samples_per_client // len(classes)
        remainder = samples_per_client % len(classes)
        class_quotas = {cls: base_quota for cls in classes}
        if remainder > 0:
            lucky_cls = np.random.choice(classes, remainder, replace=False)
            for c in lucky_cls:
                class_quotas[int(c)] += 1

        selected_indices = []

        for cls in classes:
            quota = class_quotas[cls]
            indices_of_class = class_to_indices[cls]

            if len(indices_of_class) <= quota:
                selected_indices.extend(indices_of_class)
                continue

            class_features = features[indices_of_class].cpu().numpy()

            # Group into K clusters = exact quota you need to select
            kmeans = KMeans(n_clusters=quota, random_state=42, n_init="auto")
            kmeans.fit(class_features)
            centers = kmeans.cluster_centers_

            # Find data points closest to each cluster center
            picked_local = set()
            for center in centers:
                dists = np.linalg.norm(class_features - center, axis=1)
                sorted_idx = np.argsort(dists)

                for idx_local in sorted_idx:
                    if idx_local not in picked_local:
                        picked_local.add(idx_local)
                        selected_indices.append(indices_of_class[idx_local])
                        break

        # Fill remaining quota if class-balanced k-means selected too few samples.
        # This can happen when some classes have fewer samples than their quota.
        if len(selected_indices) < samples_per_client:
            selected_set = set(selected_indices)

            for idx in range(len(labels)):
                if len(selected_indices) >= samples_per_client:
                    break

                if idx not in selected_set:
                    selected_indices.append(idx)
                    selected_set.add(idx)

        return selected_indices[:samples_per_client]

    def select_svd_kmeans_per_client(self, features, labels, samples_per_client):
        """
        Ultimate strategy for Tabular: SVD + K-Means
        1. Compress features (whether image or tabular) using SVD/PCA.
        2. Run K-Means on the compressed space (Latent Space) to get clean cluster centers.
        3. Select data points closest to the cluster centers.
        """
        from collections import defaultdict
        import numpy as np
        from sklearn.cluster import KMeans

        class_to_indices = defaultdict(list)
        for idx, lbl in enumerate(labels.tolist()):
            class_to_indices[int(lbl)].append(idx)

        classes = list(class_to_indices.keys())
        if len(classes) == 0 or samples_per_client == 0:
            return []

        base_quota = samples_per_client // len(classes)
        remainder = samples_per_client % len(classes)
        class_quotas = {cls: base_quota for cls in classes}
        if remainder > 0:
            lucky_cls = np.random.choice(classes, remainder, replace=False)
            for c in lucky_cls:
                class_quotas[int(c)] += 1

        selected_indices = []

        for cls in classes:
            quota = class_quotas[cls]
            indices_of_class = class_to_indices[cls]

            # If there is less data than the intended compression quota, just take all
            if len(indices_of_class) <= quota:
                selected_indices.extend(indices_of_class)
                continue

            class_features = features[indices_of_class]

            # ---------------- Step 1: Dimensionality Reduction SVD ----------------
            # Compute on GPU for maximum speed, retain Top K important dimensions
            # For Letter Recognition dataset, compressing to 8-16 dimensions is optimal
            latent_dim = min(16, class_features.shape[1], class_features.shape[0])

            U, S, V = torch.pca_lowrank(class_features, q=latent_dim, center=True)
            # Equivalent to: X_compressed = (X - X_mean) @ V
            class_features_compressed = (
                torch.matmul(class_features - torch.mean(class_features, dim=0), V)
                .cpu()
                .numpy()
            )

            # ---------------- Step 2: KMEANS on Latent ----------------
            kmeans = KMeans(n_clusters=quota, random_state=42, n_init="auto")
            kmeans.fit(class_features_compressed)
            centers = kmeans.cluster_centers_

            # ---------------- Step 3: Find points closest to the center ----------------
            picked_local = set()
            for center in centers:
                dists = np.linalg.norm(class_features_compressed - center, axis=1)
                sorted_idx = np.argsort(dists)

                for idx_local in sorted_idx:
                    if idx_local not in picked_local:
                        picked_local.add(idx_local)
                        selected_indices.append(indices_of_class[idx_local])
                        break

        # Fill remaining quota if class-balanced SVD+k-means selected too few samples.
        if len(selected_indices) < samples_per_client:
            selected_set = set(selected_indices)

            for idx in range(len(labels)):
                if len(selected_indices) >= samples_per_client:
                    break

                if idx not in selected_set:
                    selected_indices.append(idx)
                    selected_set.add(idx)

        return selected_indices[:samples_per_client]
