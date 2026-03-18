import torch
from torch.utils.data import DataLoader
from model import extract_features


def extract_local_features(model, dataset, device, batch_size=128):
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats = extract_features(model, images).cpu()
            all_features.append(feats)
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return features, labels


def extract_all_client_features(model, client_datasets, device, batch_size=128):
    """
    对所有 client 分别提 feature
    return:
        client_features: list of tensors
        client_labels:   list of tensors
    """
    client_features = []
    client_labels = []

    for dataset in client_datasets:
        feats, labels = extract_local_features(model, dataset, device, batch_size)
        client_features.append(feats)
        client_labels.append(labels)

    return client_features, client_labels


def generate_square_orthogonal_matrix(n):
    """
    生成 n x n 的随机正交矩阵
    """
    A = torch.randn(n, n)
    Q, _ = torch.linalg.qr(A)
    return Q


def build_pseudo_features_exact(features, target_dim=128):
    """
    真正按论文形式近似实现:
        X' = P X Q

    Args:
        features: [N, d]
        target_dim: 先把右侧 Q 压到 target_dim 维

    Returns:
        pseudo_features: [N, target_dim]
    """
    N, d = features.shape

    # 右侧 shared Q: [d, target_dim]
    Q_right = torch.randn(d, target_dim)
    Q_right, _ = torch.linalg.qr(Q_right)

    # 左侧 client-specific P: [N, N]
    P_left = generate_square_orthogonal_matrix(N)

    # XQ
    reduced = features @ Q_right  # [N, target_dim]

    # P(XQ)
    pseudo = P_left @ reduced  # [N, target_dim]

    return pseudo


def build_all_client_pseudo_features_exact(client_features, target_dim=128):
    """
    对所有 client 分别构造 exact pseudo features
    """
    client_pseudo_features = []

    for features in client_features:
        pseudo = build_pseudo_features_exact(features, target_dim=target_dim)
        client_pseudo_features.append(pseudo)

    return client_pseudo_features


def concat_client_pseudo_features(client_pseudo_features):
    """
    server 端把所有 client 的 pseudo features 拼接起来
    """
    return torch.cat(client_pseudo_features, dim=0)


def server_svd_global(global_pseudo, k=32):
    """
    对 global concatenated pseudo features 做 SVD
    返回前 k 列的 U_k_global
    """
    U, S, Vh = torch.linalg.svd(global_pseudo, full_matrices=False)
    U_k_global = U[:, :k]
    return U_k_global


def split_global_U_to_clients(U_global, client_sizes):
    """
    按 client 的样本数把全局 U 切回每个 client 对应的 U_k
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
    根据某个 client 的 U_k 计算每个样本的 leverage score

    Args:
        client_U: [N_k, k]

    Returns:
        scores: [N_k]
    """
    scores = torch.sum(client_U**2, dim=1)
    return scores


def select_replay_per_client(client_scores, client_labels, samples_per_client):
    """
    每个 client 内部，根据 leverage score 做 class-balanced 选择

    Args:
        client_scores: [N_k]
        client_labels: [N_k]
        samples_per_client: 每个 client 选多少

    Returns:
        selected_indices: list[int]
    """
    from collections import defaultdict

    class_to_items = defaultdict(list)

    for idx in range(len(client_labels)):
        cls = int(client_labels[idx].item())
        score = float(client_scores[idx].item())
        class_to_items[cls].append((idx, score))

    classes = list(class_to_items.keys())
    num_classes = len(classes)

    samples_per_class = samples_per_client // num_classes

    selected_indices = []

    for cls in classes:
        items = class_to_items[cls]

        # 按 leverage score 排序（从大到小）
        items.sort(key=lambda x: x[1], reverse=True)

        chosen = [idx for idx, _ in items[:samples_per_class]]
        selected_indices.extend(chosen)

    return selected_indices
