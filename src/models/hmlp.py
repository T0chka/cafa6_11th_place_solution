"""Hierarchy-aware multilabel MLP used by the HMLP component."""

import time
import numpy as np
import torch

from pathlib import Path
from dataclasses import dataclass
from scipy import sparse
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.training import (
    normalize_train_valid,
    apply_normalize_test,
    multilabel_metrics,
    print_fold_metrics,
)


@dataclass(frozen=True)
class TorchMLPConfig:
    hidden_dim: int = 2048
    dropout: float = 0.5
    weight_decay: float = 0.25
    max_lr: float = 0.0003
    batch_size: int = 512
    max_epochs: int = 100
    patience: int = 5
    seed: int = 42
    n_splits: int = 5
    device: str = "cuda"
    normalization: str = "zscore"
    parent_loss_weight: float = 0.25
    parent_gate_init: float = 2.0
    parent_pooling: str = "logsumexp"


class TorchMLPModel:
    def __init__(self, config: TorchMLPConfig) -> None:
        self.config = config

    @dataclass(frozen=True)
    class _LossState:
        child_mask_t: torch.Tensor
        parent_mask_t: torch.Tensor
        has_parents: bool

    def _make_loss_state(self, net: nn.Module, device: str) -> _LossState:
        n_targets = int(net.child_head.out_features)
        parent_full_indices = net.parent_full_indices.detach().cpu().numpy()
        parent_full_indices = parent_full_indices.astype(np.int64, copy=False)

        parent_mask_full = np.zeros((n_targets,), dtype=bool)
        parent_mask_full[parent_full_indices] = True
        child_mask_full = ~parent_mask_full

        parent_mask_t = torch.from_numpy(parent_mask_full).to(device)
        child_mask_t = torch.from_numpy(child_mask_full).to(device)

        return self._LossState(
            child_mask_t=child_mask_t,
            parent_mask_t=parent_mask_t,
            has_parents=bool(parent_mask_full.any()),
        )

    def _compute_hierarchical_loss(
        self,
        logits_full: torch.Tensor,
        child_logits_full: torch.Tensor,
        targets_batch: torch.Tensor,
        state: _LossState,
        bce: nn.Module,
    ) -> torch.Tensor:
        loss_child = bce(
            child_logits_full[:, state.child_mask_t],
            targets_batch[:, state.child_mask_t],
        )
        if not state.has_parents:
            return loss_child

        loss_parent = bce(
            logits_full[:, state.parent_mask_t],
            targets_batch[:, state.parent_mask_t],
        )
        return loss_child + float(self.config.parent_loss_weight) * loss_parent

    @staticmethod
    def _build_parent_edges(
        aspect_gt: dict[str, object],
        term_ids: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ontology_term_ids = np.asarray(aspect_gt["ontology_term_ids"], dtype=object)
        graph = aspect_gt["graph"]
        parents_indptr = np.asarray(graph["parents_indptr"], dtype=np.int64)
        parents_indices = np.asarray(graph["parents_indices"], dtype=np.int32)

        term_to_local = {str(t): int(i) for i, t in enumerate(term_ids)}
        ont_to_pos = {str(t): int(i) for i, t in enumerate(ontology_term_ids.tolist())}

        local_to_ont = np.fromiter(
            (ont_to_pos[str(t)] for t in term_ids),
            dtype=np.int32,
            count=int(len(term_ids)),
        )

        parent_full_indices: list[int] = []
        edge_child: list[int] = []
        edge_parent_pos: list[int] = []
        parent_pos_map: dict[int, int] = {}

        for child_local, child_ont in enumerate(local_to_ont.tolist()):
            p0 = int(parents_indptr[child_ont])
            p1 = int(parents_indptr[child_ont + 1])
            for parent_ont in parents_indices[p0:p1].tolist():
                parent_term = str(ontology_term_ids[int(parent_ont)])
                parent_local = term_to_local.get(parent_term)
                if parent_local is None:
                    continue
                parent_pos = parent_pos_map.get(parent_local)
                if parent_pos is None:
                    parent_pos = len(parent_full_indices)
                    parent_pos_map[parent_local] = parent_pos
                    parent_full_indices.append(parent_local)
                edge_child.append(int(child_local))
                edge_parent_pos.append(int(parent_pos))

        return (
            np.asarray(parent_full_indices, dtype=np.int64),
            np.asarray(edge_child, dtype=np.int64),
            np.asarray(edge_parent_pos, dtype=np.int64),
        )

    def fit(
        self,
        features: np.ndarray,
        targets: sparse.csr_matrix,
        term_ids: list[str],
        aspect_gt: dict[str, object],
        out_dir: Path,
        debug: bool = False,
        compute_metrics: bool = True,
    ) -> dict[str, object]:
        device = self.config.device

        models_dir = out_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        n_samples = int(features.shape[0])
        n_targets = int(targets.shape[1])

        parent_full_indices, edge_child, edge_parent_pos = self._build_parent_edges(
            aspect_gt=aspect_gt,
            term_ids=term_ids,
        )
        parent_full_indices_t = torch.from_numpy(parent_full_indices)
        edge_child_t = torch.from_numpy(edge_child)
        edge_parent_pos_t = torch.from_numpy(edge_parent_pos)

        if debug:
            print(f"[DEBUG] fit: n_samples={n_samples}, n_targets={n_targets}")

        targets = np.asarray(targets.toarray(), dtype=np.float32)

        folds = self._make_folds(n_samples, self.config.n_splits)
        if debug:
            print(f"[DEBUG] Created folds: {self.config.n_splits} splits")
            fold_counts = {
                i: int(np.sum(folds == i)) for i in range(1, self.config.n_splits + 1)
            }
            print(f"[DEBUG] Fold sizes: {fold_counts}")

        oof_logits = np.full((n_samples, n_targets), np.nan, dtype=np.float32)
        fold_val_loss: list[float] = []
        fold_val_auc: list[float] = []
        fold_val_ap: list[float] = []

        for fold_id in range(1, self.config.n_splits + 1):
            print(f"\nFold {fold_id}/{self.config.n_splits}")
            train_idx = np.where(folds != fold_id)[0]
            valid_idx = np.where(folds == fold_id)[0]

            x_train, x_valid, norm_params = normalize_train_valid(
                features, train_idx, valid_idx, method=self.config.normalization
            )
            if debug:
                print(
                    f"[DEBUG] Fold {fold_id}: x_train shape={x_train.shape}, "
                    f"x_valid shape={x_valid.shape}"
                )

            y_train = targets[train_idx]
            y_valid = targets[valid_idx]

            train_dl = DataLoader(
                TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                batch_size=self.config.batch_size,
                shuffle=True,
                drop_last=False,
            )
            valid_dl = DataLoader(
                TensorDataset(torch.from_numpy(x_valid), torch.from_numpy(y_valid)),
                batch_size=self.config.batch_size,
                shuffle=False,
                drop_last=False,
            )

            net = _HierMLP(
                n_features=int(features.shape[1]),
                hidden_dim=self.config.hidden_dim,
                n_targets=n_targets,
                dropout=self.config.dropout,
                parent_full_indices=parent_full_indices_t.to(device),
                edge_child=edge_child_t.to(device),
                edge_parent_pos=edge_parent_pos_t.to(device),
                parent_gate_init=self.config.parent_gate_init,
                parent_pooling=self.config.parent_pooling,
            ).to(device)

            best_state, _ = self._fit_one_fold(net, train_dl, valid_dl)

            net.load_state_dict(best_state)
            logits, val_loss = self._predict_logits_and_loss(net, valid_dl)

            if compute_metrics:
                y_valid_probs = self._sigmoid(logits)
                val_auc, val_ap, n_valid_terms = multilabel_metrics(
                    y_valid, y_valid_probs
                )
                if n_valid_terms < y_valid.shape[1]:
                    print(
                        f"[WARNING] Fold {fold_id}: valid terms for AUC/AP = "
                        f"{n_valid_terms}/{y_valid.shape[1]}"
                    )
                print(
                    f"[INFO] Fold {fold_id}: val_loss={val_loss:.4f}, "
                    f"val_auc={val_auc:.4f}, val_ap={val_ap:.4f}"
                )
                fold_val_auc.append(float(val_auc))
                fold_val_ap.append(float(val_ap))
            else:
                print(f"[INFO] Fold {fold_id}: val_loss={val_loss:.4f}")
                fold_val_auc.append(np.nan)
                fold_val_ap.append(np.nan)

            oof_logits[valid_idx] = logits
            fold_val_loss.append(float(val_loss))

            torch.save(net.state_dict(), models_dir / f"fold_{fold_id:02d}.pt")

            norm_dict = {"method": self.config.normalization}
            if self.config.normalization == "zscore":
                norm_dict["mean"] = norm_params["mean"].astype(np.float32, copy=False)
                norm_dict["std"] = norm_params["std"].astype(np.float32, copy=False)
            elif self.config.normalization == "minmax":
                norm_dict["min"] = norm_params["min"].astype(np.float32, copy=False)
                norm_dict["max"] = norm_params["max"].astype(np.float32, copy=False)
            elif self.config.normalization == "robust":
                norm_dict["median"] = norm_params["median"].astype(
                    np.float32, copy=False
                )
                norm_dict["q25"] = norm_params["q25"].astype(np.float32, copy=False)
                norm_dict["q75"] = norm_params["q75"].astype(np.float32, copy=False)

            np.savez_compressed(models_dir / f"fold_{fold_id:02d}_norm.npz", **norm_dict)

        mean_loss, mean_auc, mean_ap = print_fold_metrics(
            fold_val_loss=fold_val_loss,
            fold_val_auc=fold_val_auc if compute_metrics else None,
            fold_val_ap=fold_val_ap if compute_metrics else None,
            debug=debug,
            oof_logits=oof_logits,
        )

        # Save term_ids for test prediction
        np.save(out_dir / "term_ids.npy", np.asarray(term_ids, dtype=object))

        return {
            "model": "mlp_hier",
            "n_samples": n_samples,
            "n_targets": n_targets,
            "n_splits": int(self.config.n_splits),
            "val_loss_mean": mean_loss,
            "val_auc_mean": mean_auc,
            "val_ap_mean": mean_ap,
            "oof_logits": oof_logits,
        }

    def _make_folds(self, n_samples: int, n_splits: int) -> np.ndarray:
        rng = np.random.default_rng(self.config.seed)
        folds = rng.integers(1, n_splits + 1, size=n_samples).astype(np.int32)
        return folds

    def _sigmoid(self, logits: np.ndarray) -> np.ndarray:
        probs = 1.0 / (1.0 + np.exp(-logits))
        return probs.astype(np.float32)

    def _fit_one_fold(
        self,
        net: nn.Module,
        train_dl: DataLoader,
        valid_dl: DataLoader,
    ) -> tuple[dict | None, float]:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        device = self.config.device

        opt = torch.optim.AdamW(
            net.parameters(),
            lr=self.config.max_lr,
            weight_decay=self.config.weight_decay,
        )
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.config.max_lr,
            epochs=self.config.max_epochs,
            steps_per_epoch=len(train_dl),
        )

        bce = nn.BCEWithLogitsLoss()
        state = self._make_loss_state(net=net, device=device)

        best = float("inf")
        best_state = None
        bad = 0

        for epoch in range(1, self.config.max_epochs + 1):
            epoch_start = time.time()

            net.train()
            train_total, train_n_obs = 0.0, 0
            for features_batch, targets_batch in train_dl:
                features_batch = features_batch.to(device)
                targets_batch = targets_batch.to(device)

                opt.zero_grad(set_to_none=True)
                logits_full, child_logits_full = net(features_batch)

                loss = self._compute_hierarchical_loss(
                    logits_full=logits_full,
                    child_logits_full=child_logits_full,
                    targets_batch=targets_batch,
                    state=state,
                    bce=bce,
                )

                loss.backward()
                opt.step()
                sched.step()

                batch_size = int(features_batch.shape[0])
                train_total += float(loss.item()) * batch_size
                train_n_obs += batch_size

            train_loss = train_total / max(train_n_obs, 1)
            current_lr = float(sched.get_last_lr()[0])

            net.eval()
            val_total, val_n_obs = 0.0, 0
            with torch.no_grad():
                for features_batch, targets_batch in valid_dl:
                    features_batch = features_batch.to(device)
                    targets_batch = targets_batch.to(device)

                    logits_full, child_logits_full = net(features_batch)
                    loss = self._compute_hierarchical_loss(
                        logits_full=logits_full,
                        child_logits_full=child_logits_full,
                        targets_batch=targets_batch,
                        state=state,
                        bce=bce,
                    )

                    batch_size = int(features_batch.shape[0])
                    val_total += float(loss.item()) * batch_size
                    val_n_obs += batch_size

            cur = val_total / max(val_n_obs, 1)
            epoch_time = time.time() - epoch_start

            is_best = bool(cur < best)
            bad = 0 if is_best else (bad + 1)
            status = "✓" if is_best else " "
            print(
                f"  Epoch {epoch:3d}/{self.config.max_epochs}: "
                f"train_loss={train_loss:.4f} val_loss={cur:.4f} "
                f"lr={current_lr:.6f} time={epoch_time:.1f}s {status}"
            )

            if is_best:
                best = cur
                best_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
            elif bad >= self.config.patience:
                print(
                    f"  Early stopping at epoch {epoch} "
                    f"(patience={self.config.patience})"
                )
                break

        return best_state, best

    def _predict_logits_and_loss(
        self,
        net: nn.Module,
        valid_dl: DataLoader,
    ) -> tuple[np.ndarray, float]:
        net.eval()
        device = self.config.device

        bce = nn.BCEWithLogitsLoss()
        state = self._make_loss_state(net=net, device=device)

        logits_list: list[np.ndarray] = []
        total, n_obs = 0.0, 0

        with torch.no_grad():
            for features_batch, targets_batch in valid_dl:
                features_batch = features_batch.to(device)
                targets_batch = targets_batch.to(device)

                logits_full, child_logits_full = net(features_batch)
                loss = self._compute_hierarchical_loss(
                    logits_full=logits_full,
                    child_logits_full=child_logits_full,
                    targets_batch=targets_batch,
                    state=state,
                    bce=bce,
                )

                total += float(loss.item()) * int(features_batch.shape[0])
                n_obs += int(features_batch.shape[0])
                logits_list.append(logits_full.detach().cpu().numpy().astype(np.float32))

        logits_all = np.vstack(logits_list)
        return logits_all, total / max(n_obs, 1)

    def predict_logits_ensemble(
        self,
        features: np.ndarray,
        aspect_gt: dict[str, object],
        aspect_dir: Path,
    ) -> np.ndarray:
        device = self.config.device
        models_dir = aspect_dir / "models"
        if not models_dir.exists():
            raise FileNotFoundError(f"Missing models_dir: {models_dir}")

        term_ids_path = aspect_dir / "term_ids.npy"
        if not term_ids_path.exists():
            raise FileNotFoundError(f"Missing term_ids: {term_ids_path}")

        term_ids = np.load(term_ids_path, allow_pickle=True)

        parent_full_indices, edge_child, edge_parent_pos = self._build_parent_edges(
            aspect_gt=aspect_gt,
            term_ids=term_ids,
        )
        parent_full_indices_t = torch.from_numpy(parent_full_indices).to(device)
        edge_child_t = torch.from_numpy(edge_child).to(device)
        edge_parent_pos_t = torch.from_numpy(edge_parent_pos).to(device)

        fold_logits = None

        for fold_id in range(1, self.config.n_splits + 1):
            state_path = models_dir / f"fold_{fold_id:02d}.pt"
            norm_path = models_dir / f"fold_{fold_id:02d}_norm.npz"
            if (not state_path.exists()) or (not norm_path.exists()):
                raise FileNotFoundError(f"Missing fold artifacts for fold {fold_id}")

            norm = np.load(norm_path)
            method = str(norm["method"]) if "method" in norm else "zscore"
            features_fold = apply_normalize_test(features, norm, method)

            net = _HierMLP(
                n_features=int(features.shape[1]),
                hidden_dim=self.config.hidden_dim,
                n_targets=len(term_ids),
                dropout=self.config.dropout,
                parent_full_indices=parent_full_indices_t.to(device),
                edge_child=edge_child_t.to(device),
                edge_parent_pos=edge_parent_pos_t.to(device),
                parent_gate_init=self.config.parent_gate_init,
                parent_pooling=self.config.parent_pooling,
            ).to(device)

            state = torch.load(state_path, map_location="cpu")
            net.load_state_dict(state)
            net.eval()

            dl = DataLoader(
                TensorDataset(torch.from_numpy(features_fold)),
                batch_size=self.config.batch_size,
                shuffle=False,
                drop_last=False,
            )

            logits_list: list[np.ndarray] = []
            with torch.no_grad():
                for (features_batch,) in dl:
                    features_batch = features_batch.to(device)
                    logits_full, _ = net(features_batch)
                    logits_list.append(
                        logits_full.detach().cpu().numpy().astype(np.float32)
                    )

            logits_all = np.vstack(logits_list)
            fold_logits = logits_all if fold_logits is None else fold_logits + logits_all

        return (fold_logits / float(self.config.n_splits)).astype(np.float32, copy=False)


class _HierMLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        n_targets: int,
        dropout: float,
        parent_full_indices: torch.Tensor,
        edge_child: torch.Tensor,
        edge_parent_pos: torch.Tensor,
        parent_gate_init: float,
        parent_pooling: str = "amax",
        pooling_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pooling_eps = float(pooling_eps)
        self.child_head = nn.Linear(hidden_dim, n_targets)

        self.parent_full_indices = parent_full_indices
        self.edge_child = edge_child
        self.edge_parent_pos = edge_parent_pos

        n_parents = int(parent_full_indices.numel())
        self.parent_head = nn.Linear(hidden_dim, n_parents) if n_parents else None
        self.parent_gate_logit = nn.Parameter(
            torch.full((n_parents,), float(parent_gate_init))
        )

        self.parent_pooling = str(parent_pooling)

    def _pool_parents_from_children(
        self,
        child_logits_full: torch.Tensor,
        batch_size: int,
        n_parents: int,
    ) -> torch.Tensor:
        child_edge_logits = child_logits_full[:, self.edge_child]
        parent_index = self.edge_parent_pos.expand(batch_size, -1)

        if self.parent_pooling == "amax":
            parent_vals = torch.full(
                (batch_size, n_parents),
                -torch.inf,
                device=child_logits_full.device,
                dtype=child_logits_full.dtype,
            )
            return parent_vals.scatter_reduce(
                1,
                parent_index,
                child_edge_logits,
                reduce="amax",
                include_self=True,
            )

        if self.parent_pooling == "logsumexp":
            parent_max = torch.full(
                (batch_size, n_parents),
                -torch.inf,
                device=child_logits_full.device,
                dtype=child_logits_full.dtype,
            )
            parent_max = parent_max.scatter_reduce(
                1,
                parent_index,
                child_edge_logits,
                reduce="amax",
                include_self=True,
            )
            max_per_edge = parent_max.gather(1, parent_index)
            exp_shifted = torch.exp(child_edge_logits - max_per_edge)
            sum_exp = torch.zeros(
                (batch_size, n_parents),
                device=child_logits_full.device,
                dtype=child_logits_full.dtype,
            )
            sum_exp.scatter_add_(1, parent_index, exp_shifted)
            return parent_max + torch.log(sum_exp.clamp_min(self.pooling_eps))

        if self.parent_pooling == "noisy_or":
            prob_child = torch.sigmoid(child_edge_logits)
            prob_child = prob_child.clamp(self.pooling_eps, 1.0 - self.pooling_eps)
            log_comp = torch.log1p(-prob_child)
            log_prod = torch.zeros(
                (batch_size, n_parents),
                device=child_logits_full.device,
                dtype=child_logits_full.dtype,
            )
            log_prod.scatter_add_(1, parent_index, log_comp)
            prob_parent = 1.0 - torch.exp(log_prod)
            prob_parent = prob_parent.clamp(self.pooling_eps, 1.0 - self.pooling_eps)
            return torch.log(prob_parent) - torch.log1p(-prob_parent)

        raise ValueError(f"Unknown parent_pooling={self.parent_pooling!r}")

    def forward(self, features_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.trunk(features_batch)
        child_logits_full = self.child_head(representation)
        logits_full = child_logits_full

        n_parents = int(self.parent_full_indices.numel())
        if n_parents == 0:
            return logits_full, child_logits_full

        parent_direct = self.parent_head(representation)
        batch_size = int(features_batch.shape[0])
        parent_from_children = self._pool_parents_from_children(
            child_logits_full=child_logits_full,
            batch_size=batch_size,
            n_parents=n_parents,
        )
        gate = torch.sigmoid(self.parent_gate_logit).unsqueeze(0)
        parent_structural = gate * parent_from_children
        parent_logits = torch.maximum(parent_direct, parent_structural)

        logits_full = logits_full.clone()
        logits_full[:, self.parent_full_indices] = parent_logits
        return logits_full, child_logits_full

HMLPConfig = TorchMLPConfig
HMLPModel = TorchMLPModel
