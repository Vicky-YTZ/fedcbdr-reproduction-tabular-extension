import torch
from torch.utils.data import DataLoader
from model import extract_features


def extract_local_features(model, dataset, device, batch_size=128, extract_fn=extract_features):
    if len(dataset) == 0:
        return torch.tensor([]), torch.tensor([])
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            feats = extract_fn(model, inputs).cpu()
            all_features.append(feats)
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return features, labels


def extract_all_client_features(model, client_datasets, device, batch_size=128, extract_fn=extract_features):
    """
    Extract features for all clients separately.
    
    Returns:
        client_features: list of tensors
        client_labels:   list of tensors
    """
    client_features = []
    client_labels = []

    for dataset in client_datasets:
        if len(dataset) == 0:
            client_features.append(torch.tensor([]))
            client_labels.append(torch.tensor([]))
        else:
            feats, labels = extract_local_features(model, dataset, device, batch_size, extract_fn=extract_fn)
            client_features.append(feats)
            client_labels.append(labels)

    return client_features, client_labels


def generate_square_orthogonal_matrix(n):
    """
    Generate an n x n random orthogonal matrix.
    """
    A = torch.randn(n, n)
    Q, _ = torch.linalg.qr(A)
    return Q


def build_pseudo_features_exact(features, target_dim=128):
    """
    Exact implementation according to the paper format:
        X' = P X Q

    Args:
        features: [N, d]
        target_dim: Dimension to compress the right-side matrix Q down to.

    Returns:
        pseudo_features: [N, target_dim]
    """
    N, d = features.shape

    # Right-side shared Q: [d, target_dim]
    Q_right = torch.randn(d, target_dim)
    Q_right, _ = torch.linalg.qr(Q_right)

    # Left-side client-specific P: [N, N]
    P_left = generate_square_orthogonal_matrix(N)

    # XQ
    reduced = features @ Q_right  # [N, target_dim]

    # P(XQ)
    pseudo = P_left @ reduced  # [N, target_dim]

    return pseudo


def build_all_client_pseudo_features_exact(client_features, target_dim=128):
    """
    Construct exact pseudo features for all clients separately.
    """
    client_pseudo_features = []

    for features in client_features:
        if features.numel() == 0:
            pseudo = torch.empty(0, target_dim)
        else:
            pseudo = build_pseudo_features_exact(features, target_dim=target_dim)
        client_pseudo_features.append(pseudo)

    return client_pseudo_features


def concat_client_pseudo_features(client_pseudo_features):
    """
    Concatenate all clients' pseudo features on the server side.
    """
    return torch.cat(client_pseudo_features, dim=0)


def server_svd_global(global_pseudo, k=32):
    """
    Perform SVD on the globally concatenated pseudo features.
    Returns the top k columns of U_global.
    """
    U, S, Vh = torch.linalg.svd(global_pseudo, full_matrices=False)
    U_k_global = U[:, :k]
    return U_k_global


def split_global_U_to_clients(U_global, client_sizes):
    """
    Split the global U matrix back to the corresponding U_k for each client based on their sample counts.
    """
    client_U_list = []

    start = 0
    for size in client_sizes:
        end = start + size
        client_U_list.append(U_global[start:end])
        start = end

    return client_U_list


def compute_client_leverage_scores(client_U):
    """
    Calculate the leverage score for each sample based on a client's U_k.

    Args:
        client_U: [N_k, k]

    Returns:
        scores: [N_k]
    """
    scores = torch.sum(client_U**2, dim=1)
    return scores


def select_replay_per_client(client_scores, client_labels, samples_per_client):
    from collections import defaultdict
    import torch
    import numpy as np

    class_to_items = defaultdict(list)

    # Group samples by their labels alongside their GDR leverage scores
    for idx in range(len(client_labels)):
        cls = int(client_labels[idx].item())
        score = float(client_scores[idx].item())
        class_to_items[cls].append((idx, score))

    classes = list(class_to_items.keys())
    if len(classes) == 0 or samples_per_client == 0:
        return []

    # 1. Start with an even distribution of quota across the available classes
    base_quota = int(samples_per_client // len(classes))
    remainder = int(samples_per_client % len(classes))
    
    class_quotas = {cls: base_quota for cls in classes}
    
    # Randomly distribute the remainder to avoid biasing the first class
    if remainder > 0:
        lucky_classes = np.random.choice(classes, remainder, replace=False)
        for c in lucky_classes:
            class_quotas[int(c)] += 1

    selected_indices = []

    # 2. First pass: Collect samples. If a class has fewer samples than its quota, 
    # take all of them and accumulate the unused quota to redistribute.
    leftover_quota = 0
    active_classes = [] # Classes that have more samples than their current quota

    for cls in classes:
        items = class_to_items[cls]
        quota = class_quotas[cls]
        
        if len(items) <= quota:
            # Not enough samples or just enough: take all and save the remainder
            selected_indices.extend([int(x[0]) for x in items])
            leftover_quota += (quota - len(items))
            class_quotas[cls] = 0 # Reset quota
        else:
            # We have more samples than quota
            active_classes.append(cls)

    # 3. Redistribute leftover quota evenly among active_classes (if any)
    while leftover_quota > 0 and len(active_classes) > 0:
        bonus = int(leftover_quota // len(active_classes))
        bonus_rem = int(leftover_quota % len(active_classes))
        
        leftover_quota = 0 # reset for the next potential iteration
        new_active = []
        
        lucky_bonus = np.random.choice(active_classes, bonus_rem, replace=False) if bonus_rem > 0 else []
        lucky_bonus_set = set(int(x) for x in lucky_bonus)
        
        for cls in active_classes:
            added_quota = bonus + (1 if cls in lucky_bonus_set else 0)
            class_quotas[cls] += added_quota
            
            # Check if the new quota exceeds available items
            items = class_to_items[cls]
            if len(items) <= class_quotas[cls]:
                selected_indices.extend([int(x[0]) for x in items])
                leftover_quota += (class_quotas[cls] - len(items))
                class_quotas[cls] = 0 # Mark as processed
            else:
                new_active.append(cls)
                
        active_classes = new_active

    # 4. Final pass: Sample based on GDR Leverage Scores for classes that still have items to pick
    for cls in classes:
        final_quota = class_quotas[cls]
        if final_quota == 0:
            continue
            
        items = class_to_items[cls]
        indices = [int(x[0]) for x in items]
        scores = torch.tensor([x[1] for x in items], dtype=torch.float32)

        # Build probability distribution based on GDR scores
        if scores.sum() > 0:
            probs = scores / scores.sum()
        else:
            probs = torch.ones_like(scores) / len(scores)
            
        probs = probs.nan_to_num(1.0/len(scores))
        
        c_samples = int(min(final_quota, len(probs)))
        sampled_idxs = torch.multinomial(probs, num_samples=c_samples, replacement=False)
        
        selected_indices.extend([indices[int(i.item())] for i in sampled_idxs])

    return selected_indices