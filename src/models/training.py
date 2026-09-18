"""
Helper functions for training models.
"""

import numpy as np
import numba as nb


@nb.njit(cache=True)
def _auc_from_sorted(scores: np.ndarray, labels: np.ndarray,
                     order: np.ndarray) -> float:
    n = int(labels.size)
    n_pos = 0
    for i in range(n):
        n_pos += int(labels[i])
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan

    sum_ranks_pos = 0.0
    rank = 1
    i = 0
    while i < n:
        s = scores[order[i]]
        j = i + 1
        pos_in_group = int(labels[order[i]])
        while j < n and scores[order[j]] == s:
            pos_in_group += int(labels[order[j]])
            j += 1
        group_len = j - i
        avg_rank = (rank + (rank + group_len - 1)) * 0.5
        sum_ranks_pos += avg_rank * pos_in_group
        rank += group_len
        i = j

    baseline = n_pos * (n_pos + 1) * 0.5
    return (sum_ranks_pos - baseline) / (n_pos * n_neg)


@nb.njit(cache=True)
def _ap_from_sorted_desc(scores: np.ndarray, labels: np.ndarray,
                         order: np.ndarray) -> float:
    n = int(labels.size)
    n_pos = 0
    for i in range(n):
        n_pos += int(labels[i])
    if n_pos == 0:
        return np.nan

    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0

    p = n - 1
    while p >= 0:
        s = scores[order[p]]
        tp_add = 0
        fp_add = 0
        while p >= 0 and scores[order[p]] == s:
            if labels[order[p]] == 1:
                tp_add += 1
            else:
                fp_add += 1
            p -= 1
        tp += tp_add
        fp += fp_add
        precision = tp / (tp + fp)
        recall = tp / n_pos
        ap += precision * (recall - prev_recall)
        prev_recall = recall

    return ap


@nb.njit(parallel=True, cache=True)
def _auc_ap_per_label(y_true: np.ndarray, y_prob: np.ndarray,
                      label_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = int(label_indices.size)
    auc_vec = np.empty(m, dtype=np.float64)
    ap_vec = np.empty(m, dtype=np.float64)

    for k in nb.prange(m):
        j = int(label_indices[k])
        scores = y_prob[:, j]
        labels = y_true[:, j]
        order = np.argsort(scores)
        auc_vec[k] = _auc_from_sorted(scores, labels, order)
        ap_vec[k] = _ap_from_sorted_desc(scores, labels, order)

    return auc_vec, ap_vec


def multilabel_metrics(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[float, float, int]:
    y_true = np.asarray(y_true, dtype=np.uint8, order="C")
    y_prob = np.asarray(y_prob, dtype=np.float32, order="C")
    n_rows = int(y_true.shape[0])

    pos = y_true.sum(axis=0)
    valid = (pos > 0) & (pos < n_rows)
    label_indices = np.flatnonzero(valid).astype(np.int64, copy=False)
    n_valid = int(label_indices.size)
    if n_valid == 0:
        return float("nan"), float("nan"), 0

    auc_vec, ap_vec = _auc_ap_per_label(y_true, y_prob, label_indices)
    return float(np.nanmean(auc_vec)), float(np.nanmean(ap_vec)), n_valid


def normalize_train_valid(
        x: np.ndarray,
        train_idx: np.ndarray,
        valid_idx: np.ndarray,
        method: str = "zscore",
        eps: float = 1e-6
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Normalize training and validation data using specified method.

        Methods:
        - "zscore": Standardization (mean=0, std=1) - default, good for most cases
        - "minmax": Min-Max scaling to [0, 1] - good when data has bounded range
        - "robust": Robust scaling using median and IQR - good for data with outliers
        - "l2": L2 normalization (unit vectors) - good for embeddings, preserves direction
        - "none": No normalization - use raw features
        """
        x_train_raw = x[train_idx].astype(np.float32, copy=False)
        x_valid_raw = x[valid_idx].astype(np.float32, copy=False)

        if method == "zscore":
            # Standardization: (x - mean) / std
            mean = x_train_raw.mean(axis=0, dtype=np.float64)
            std = x_train_raw.std(axis=0, dtype=np.float64)
            std = np.where(std < eps, 1.0, std)
            mean = mean.astype(np.float32, copy=False)
            std = std.astype(np.float32, copy=False)
            x_train = ((x_train_raw - mean) / std).astype(np.float32, copy=False)
            x_valid = ((x_valid_raw - mean) / std).astype(np.float32, copy=False)
            return x_train, x_valid, {"mean": mean, "std": std}

        elif method == "minmax":
            # Min-Max scaling: (x - min) / (max - min)
            x_min = x_train_raw.min(axis=0)
            x_max = x_train_raw.max(axis=0)
            x_range = x_max - x_min
            x_range = np.where(x_range < eps, 1.0, x_range)
            x_train = ((x_train_raw - x_min) / x_range).astype(np.float32, copy=False)
            x_valid = ((x_valid_raw - x_min) / x_range).astype(np.float32, copy=False)
            return x_train, x_valid, {"min": x_min, "max": x_max}

        elif method == "robust":
            # Robust scaling: (x - median) / IQR
            # Uses median and interquartile range (IQR = Q75 - Q25)
            median = np.median(x_train_raw, axis=0).astype(np.float32, copy=False)
            q25 = np.percentile(x_train_raw, 25, axis=0).astype(np.float32, copy=False)
            q75 = np.percentile(x_train_raw, 75, axis=0).astype(np.float32, copy=False)
            iqr = q75 - q25
            iqr = np.where(iqr < eps, 1.0, iqr)
            x_train = ((x_train_raw - median) / iqr).astype(np.float32, copy=False)
            x_valid = ((x_valid_raw - median) / iqr).astype(np.float32, copy=False)
            return x_train, x_valid, {"median": median, "q25": q25, "q75": q75}

        elif method == "l2":
            # L2 normalization: x / ||x||_2 (unit vectors)
            # Normalize each sample to have unit L2 norm
            train_norms = np.linalg.norm(x_train_raw, axis=1, keepdims=True)
            train_norms = np.where(train_norms < eps, 1.0, train_norms)
            x_train = (x_train_raw / train_norms).astype(np.float32, copy=False)

            # For validation, we could use train norms or compute separately
            # Using train norms for consistency (though not ideal)
            valid_norms = np.linalg.norm(x_valid_raw, axis=1, keepdims=True)
            valid_norms = np.where(valid_norms < eps, 1.0, valid_norms)
            x_valid = (x_valid_raw / valid_norms).astype(np.float32, copy=False)
            return x_train, x_valid, {}

        elif method == "none":
            # No normalization
            return x_train_raw, x_valid_raw, {}

        else:
            raise ValueError(f"Unknown normalization method: {method}")


def apply_normalize_test(
    features: np.ndarray,
    norm_params: dict,
    method: str = "zscore",
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Apply normalization to test features using saved normalization parameters.
    """
    features = features.astype(np.float32, copy=False)

    if method == "zscore":
        mean = norm_params["mean"].astype(np.float32, copy=False)
        std = norm_params["std"].astype(np.float32, copy=False)
        std = np.where(std < eps, 1.0, std)
        return ((features - mean) / std).astype(np.float32, copy=False)

    elif method == "minmax":
        min_v = norm_params["min"].astype(np.float32, copy=False)
        max_v = norm_params["max"].astype(np.float32, copy=False)
        x_range = max_v - min_v
        x_range = np.where(x_range < eps, 1.0, x_range)
        return ((features - min_v) / x_range).astype(np.float32, copy=False)

    elif method == "robust":
        median = norm_params["median"].astype(np.float32, copy=False)
        q25 = norm_params["q25"].astype(np.float32, copy=False)
        q75 = norm_params["q75"].astype(np.float32, copy=False)
        iqr = q75 - q25
        iqr = np.where(iqr < eps, 1.0, iqr)
        return ((features - median) / iqr).astype(np.float32, copy=False)

    elif method == "l2":
        norms = np.linalg.norm(features, axis=1, keepdims=True).astype(np.float32)
        norms = np.where(norms < eps, 1.0, norms)
        return (features / norms).astype(np.float32, copy=False)

    elif method == "none":
        return features.astype(np.float32, copy=False)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def print_fold_metrics(
    fold_val_loss: list[float],
    fold_val_auc: list[float] | None = None,
    fold_val_ap: list[float] | None = None,
    debug: bool = False,
    oof_logits: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """
    Print fold metrics and return mean values.
    """
    mean_loss = float(np.nanmean(np.array(fold_val_loss, dtype=np.float64)))

    if fold_val_auc is not None and fold_val_ap is not None:
        mean_auc = float(np.nanmean(np.array(fold_val_auc, dtype=np.float64)))
        mean_ap = float(np.nanmean(np.array(fold_val_ap, dtype=np.float64)))
        print(f"\n[INFO] Fold val_loss: {[f'{x:.4f}' for x in fold_val_loss]}")
        print(f"[INFO] Fold val_auc: {[f'{x:.4f}' for x in fold_val_auc]}")
        print(f"[INFO] Fold val_ap: {[f'{x:.4f}' for x in fold_val_ap]}")
        print(f"\n[INFO] Mean val_loss: {mean_loss:.4f}, Mean val_auc: {mean_auc:.4f}, Mean val_ap: {mean_ap:.4f}")
    else:
        mean_auc = np.nan
        mean_ap = np.nan
        print(f"\n[INFO] Fold val_loss: {[f'{x:.4f}' for x in fold_val_loss]}")
        print(f"\n[INFO] Mean val_loss: {mean_loss:.4f}")

    if debug and oof_logits is not None:
        print(
            f"[DEBUG] OOF logits: shape={oof_logits.shape} | "
            f"min={oof_logits.min():.4f}, max={oof_logits.max():.4f}, mean={oof_logits.mean():.4f}"
        )

    return mean_loss, mean_auc, mean_ap
