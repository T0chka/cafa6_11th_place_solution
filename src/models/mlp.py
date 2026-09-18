"""Multilabel MLP used by the ProtT5 + ESM1b component."""

import time
import numpy as np
import torch

from pathlib import Path
from dataclasses import dataclass
from scipy import sparse
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.training import (
    apply_normalize_test,
    normalize_train_valid,
    multilabel_metrics,
    print_fold_metrics,
)

@dataclass(frozen=True)
class TorchMLPConfig:
    hidden_dim: int = 2048
    n_blocks: int = 1
    layer_norm: bool = False
    narrow_layers: bool = False
    dropout: float = 0.6

    weight_decay: float = 0.2
    max_lr: float = 0.001
    batch_size: int = 512
    max_epochs: int = 200
    patience: int = 5

    seed: int = 2041
    n_splits: int = 5
    device: str = "cuda"
    normalization: str = "zscore"  # Options: "zscore", "minmax", "robust", "l2", "none"
    scheduler: str = "plateau"  # "onecycle", "cosine", "plateau"


class TorchMLPModel:
    def __init__(self, config: TorchMLPConfig) -> None:
        self.config = config

    def fit(
        self,
        features: np.ndarray,
        targets: sparse.csr_matrix,
        term_ids: list[str],
        aspect_gt: dict[str, object],
        out_dir: Path,
        debug: bool = False,
        compute_metrics: bool = True
    ) -> dict[str, object]:
        device = self.config.device

        models_dir = out_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        n_samples = int(features.shape[0])
        n_targets = int(targets.shape[1])

        if debug:
            print(f"[DEBUG] fit: n_samples={n_samples}, n_targets={n_targets}")

        # Convert sparse matrix to dense array
        targets = np.asarray(targets.toarray(), dtype=np.float32)

        folds = self._make_folds(n_samples, self.config.n_splits)
        if debug:
            print(f"[DEBUG] Created folds: {self.config.n_splits} splits")
            fold_counts = {i: int(np.sum(folds == i)) for i in range(1, self.config.n_splits + 1)}
            print(f"[DEBUG] Fold sizes: {fold_counts}")

        oof_logits = np.full((n_samples, n_targets), np.nan, dtype=np.float32)
        fold_val_loss = []
        fold_val_auc = []
        fold_val_ap = []

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
                TensorDataset(
                    torch.from_numpy(x_train),
                    torch.from_numpy(y_train),
                ),
                batch_size=self.config.batch_size,
                shuffle=True,
                drop_last=False,
            )
            valid_dl = DataLoader(
                TensorDataset(
                    torch.from_numpy(x_valid),
                    torch.from_numpy(y_valid),
                ),
                batch_size=self.config.batch_size,
                shuffle=False,
                drop_last=False,
            )

            net = _MLP(
                n_features=int(features.shape[1]),
                hidden_dim=self.config.hidden_dim,
                n_blocks=self.config.n_blocks,
                layer_norm=self.config.layer_norm,
                n_targets=n_targets,
                dropout=self.config.dropout,
                narrow_layers=self.config.narrow_layers,
            ).to(device)

            best_state, best_loss = self._fit_one_fold(net, train_dl, valid_dl)

            net.load_state_dict(best_state)
            logits, val_loss = self._predict_logits_and_loss(net, valid_dl)

            # Calculate ROC AUC and Average Precision for multilabel classification (optional)
            if compute_metrics:
                y_valid_probs = self._sigmoid(logits)
                val_auc, val_ap, n_valid_terms = multilabel_metrics(y_valid, y_valid_probs)
                if n_valid_terms < y_valid.shape[1]:
                    print(
                        f"[WARNING] Fold {fold_id}: valid terms for AUC/AP = "
                        f"{n_valid_terms}/{y_valid.shape[1]}"
                    )
                print(
                    f"[INFO] Fold {fold_id}: "
                    f"val_loss={val_loss:.4f}, val_auc={val_auc:.4f}, val_ap={val_ap:.4f}"
                )
                fold_val_auc.append(float(val_auc))
                fold_val_ap.append(float(val_ap))
            else:
                val_auc = np.nan
                val_ap = np.nan
                print(
                    f"[INFO] Fold {fold_id}: "
                    f"val_loss={val_loss:.4f}"
                )
                fold_val_auc.append(np.nan)
                fold_val_ap.append(np.nan)

            oof_logits[valid_idx] = logits

            fold_val_loss.append(float(val_loss))

            # Save model and normalization
            torch.save(net.state_dict(), models_dir / f"fold_{fold_id:02d}.pt")
            # Save normalization parameters based on method
            norm_dict = {"method": self.config.normalization}
            if self.config.normalization == "zscore":
                norm_dict["mean"] = norm_params["mean"].astype(np.float32, copy=False)
                norm_dict["std"] = norm_params["std"].astype(np.float32, copy=False)
            elif self.config.normalization == "minmax":
                norm_dict["min"] = norm_params["min"].astype(np.float32, copy=False)
                norm_dict["max"] = norm_params["max"].astype(np.float32, copy=False)
            elif self.config.normalization == "robust":
                norm_dict["median"] = norm_params["median"].astype(np.float32, copy=False)
                norm_dict["q25"] = norm_params["q25"].astype(np.float32, copy=False)
                norm_dict["q75"] = norm_params["q75"].astype(np.float32, copy=False)
            elif self.config.normalization == "l2":
                # L2 normalization doesn't need parameters, but save for consistency
                pass
            # "none" doesn't need parameters

            np.savez_compressed(
                models_dir / f"fold_{fold_id:02d}_norm.npz",
                **norm_dict
            )

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
            "model": "mlp",
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

        # Use appropriate scheduler
        if self.config.scheduler == "onecycle":
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=self.config.max_lr,
                epochs=self.config.max_epochs,
                steps_per_epoch=len(train_dl),
            )
        elif self.config.scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=self.config.max_epochs,
                eta_min=self.config.max_lr * 0.01,
            )
        elif self.config.scheduler == "plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode='min',
                factor=0.5,
                patience=self.config.patience - 3,
                min_lr=self.config.max_lr * 0.01,
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

        loss_fn = nn.BCEWithLogitsLoss()

        best = float("inf")
        best_state = None
        bad = 0

        for epoch in range(1, self.config.max_epochs + 1):
            epoch_start = time.time()

            # Training
            net.train()
            train_total, train_n_obs = 0.0, 0
            for features_batch, targets_batch in train_dl:
                features_batch = features_batch.to(device)
                targets_batch = targets_batch.to(device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(net(features_batch), targets_batch)
                loss.backward()
                opt.step()
                # Step scheduler based on type
                if self.config.scheduler == "onecycle":
                    sched.step()
                train_total += loss.item() * int(features_batch.shape[0])
                train_n_obs += int(features_batch.shape[0])
            train_loss = train_total / max(train_n_obs, 1)

            # Validation
            net.eval()
            val_total, val_n_obs = 0.0, 0
            with torch.no_grad():
                for features_batch, targets_batch in valid_dl:
                    features_batch = features_batch.to(device)
                    targets_batch = targets_batch.to(device)
                    val = loss_fn(net(features_batch), targets_batch).item()
                    val_total += val * int(features_batch.shape[0])
                    val_n_obs += int(features_batch.shape[0])
            cur = val_total / max(val_n_obs, 1)
            epoch_time = time.time() - epoch_start

            # Step scheduler for cosine/plateau
            if self.config.scheduler == "cosine":
                sched.step()
            elif self.config.scheduler == "plateau":
                sched.step(cur)

            current_lr = opt.param_groups[0]['lr'] if self.config.scheduler != "onecycle" else sched.get_last_lr()[0]

            # Print epoch info
            status = "✓" if cur < best else " "
            print(f"  Epoch {epoch:3d}/{self.config.max_epochs}: "
                  f"train_loss={train_loss:.4f} val_loss={cur:.4f} "
                  f"lr={current_lr:.6f} time={epoch_time:.1f}s {status}")

            if cur < best:
                best = cur
                best_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.config.patience:
                    print(f"  Early stopping at epoch {epoch} (patience={self.config.patience})")
                    break

        return best_state, best

    def _predict_logits_and_loss(self, net: nn.Module, valid_dl: DataLoader) -> tuple[np.ndarray, float]:
        net.eval()
        device = self.config.device

        loss_fn = nn.BCEWithLogitsLoss()
        logits_list = []
        total, n_obs = 0.0, 0
        with torch.no_grad():
            for features_batch, targets_batch in valid_dl:
                features_batch = features_batch.to(device)
                targets_batch = targets_batch.to(device)
                logits = net(features_batch)
                val = loss_fn(logits, targets_batch).item()
                total += val * int(features_batch.shape[0])
                n_obs += int(features_batch.shape[0])
                logits_list.append(logits.detach().cpu().numpy().astype(np.float32))
        logits_all = np.vstack(logits_list)
        return logits_all, total / max(n_obs, 1)

    def predict_logits_ensemble(
        self,
        features: np.ndarray,
        aspect_dir: Path,
        aspect_gt: dict[str, object] = None,
    ) -> np.ndarray:
        device = self.config.device
        models_dir = aspect_dir / "models"
        if not models_dir.exists():
            raise FileNotFoundError(f"Missing models_dir: {models_dir}")

        term_ids_path = aspect_dir / "term_ids.npy"
        if not term_ids_path.exists():
            raise FileNotFoundError(f"Missing term_ids: {term_ids_path}")

        term_ids = np.load(term_ids_path, allow_pickle=True)

        fold_logits = None

        for fold_id in range(1, self.config.n_splits + 1):
            state_path = models_dir / f"fold_{fold_id:02d}.pt"
            norm_path = models_dir / f"fold_{fold_id:02d}_norm.npz"
            if (not state_path.exists()) or (not norm_path.exists()):
                raise FileNotFoundError(f"Missing fold artifacts for fold {fold_id}")

            norm = np.load(norm_path)
            method = str(norm["method"]) if "method" in norm else "zscore"

            # Apply normalization fitted on the fold train data
            x_fold = apply_normalize_test(features, norm, method)

            state = torch.load(state_path, map_location="cpu")

            net = _MLP(
                n_features=int(features.shape[1]),
                hidden_dim=self.config.hidden_dim,
                n_blocks=self.config.n_blocks,
                layer_norm=self.config.layer_norm,
                n_targets=len(term_ids),
                dropout=self.config.dropout,
                narrow_layers=self.config.narrow_layers,
            ).to(device)
            net.load_state_dict(state)
            net.eval()

            dl = DataLoader(
                TensorDataset(torch.from_numpy(x_fold)),
                batch_size=self.config.batch_size,
                shuffle=False,
                drop_last=False,
            )

            logits_list = []
            with torch.no_grad():
                for (features_batch,) in dl:
                    features_batch = features_batch.to(device)
                    logits = net(features_batch)
                    logits_list.append(logits.detach().cpu().numpy().astype(np.float32))
            logits_all = np.vstack(logits_list)

            if fold_logits is None:
                fold_logits = logits_all
            else:
                fold_logits += logits_all

        return (fold_logits / float(self.config.n_splits)).astype(np.float32, copy=False)


class Residual(nn.Module):
    """Residual connection: features + fn(features)."""

    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.fn(features)


class MLPBlock(nn.Module):
    """Linear -> activation -> (LayerNorm) -> (Dropout)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_norm: bool = True,
        dropout: float = 0.1,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.activation = activation()
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.activation(self.linear(features))
        if self.layer_norm is not None:
            features = self.layer_norm(features)
        if self.dropout is not None:
            features = self.dropout(features)
        return features


class _MLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        n_blocks: int,
        n_targets: int,
        dropout: float,
        layer_norm: bool = True,
        narrow_layers: bool = False,
    ) -> None:
        super().__init__()
        if narrow_layers:
            nodes = [hidden_dim // (2 ** i) for i in range(n_blocks)]
        else:
            nodes = [hidden_dim for _ in range(n_blocks)]

        net: list[nn.Module] = []
        in_dim = n_features
        for out_dim in nodes:
            net.append(MLPBlock(in_dim, out_dim, dropout=dropout, layer_norm=layer_norm))
            net.append(Residual(MLPBlock(out_dim, out_dim, dropout=dropout, layer_norm=layer_norm)))
            in_dim = out_dim
        net.append(nn.Linear(in_dim, n_targets))
        self.net = nn.Sequential(*net)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

MLPConfig = TorchMLPConfig
MLPModel = TorchMLPModel
