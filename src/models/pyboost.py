"""PyBoost multilabel model used by the ProtT5 component."""

import gc
import gzip
import numpy as np
import pickle

from pathlib import Path
from dataclasses import dataclass
from scipy import sparse

from py_boost import SketchBoost
import cupy as cp
from py_boost.gpu.losses import BCELoss

from src.models.training import (
    print_fold_metrics,
    normalize_train_valid,
    apply_normalize_test,
    multilabel_metrics,
)

@dataclass(frozen=True)
class PyBoostConfig:
    ntrees: int = 5000
    lr: float = 0.03
    max_depth: int = 6
    lambda_l2: float = 1.0
    es: int = 50
    gd_steps: int = 1
    subsample: float = 0.8
    colsample: float = 0.8
    min_data_in_leaf: int = 5
    use_hess: bool = True
    sketch_method: str | None = "proj"
    sketch_outputs: int = 1
    sketch_size: int = 64
    max_bin: int = 256
    seed: int = 222
    n_splits: int = 5
    verbose: int = 200
    normalization: str = "none"  # Options: "zscore", "minmax", "robust", "l2", "none"
    term_neg_weight_alpha: float = 0
    pred_batch_size: int = 60_000  # Batch size for prediction to avoid memory issues


class BCEWithNegWeightsLoss(BCELoss):
    def __init__(self, neg_weight: cp.ndarray | None = None):
        super().__init__()
        self.neg_weight = neg_weight

    def get_grad_hess(self, y_true, y_pred):
        grad, hess = super().get_grad_hess(y_true, y_pred)
        if self.neg_weight is None:
            return grad, hess

        is_neg = (y_true <= 0.0).astype(cp.float32)
        w_neg = self.neg_weight.reshape((1, -1))
        w = 1.0 + is_neg * (w_neg - 1.0)
        return grad * w, hess * w


class PyBoostModel:
    def __init__(self, config: PyBoostConfig) -> None:
        self.config = config

    @staticmethod
    def _term_neg_weights_from_targets(
        targets: sparse.csr_matrix,
        alpha: float = 1.0,
    ) -> np.ndarray:
        n_targets = int(targets.shape[1])
        if float(alpha) <= 0.0:
            return np.ones(n_targets, dtype=np.float32)

        idx = targets.indices.astype(np.int64, copy=False)
        counts = np.bincount(idx, minlength=n_targets).astype(np.float32, copy=False)

        x = np.log1p(counts)
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        scale = float(mad if mad > 1e-6 else 1.0)
        z = (x - med) / scale

        logw = float(alpha) * np.tanh(z)
        w = np.exp(logw).astype(np.float32, copy=False)
        w /= float(np.mean(w))
        return w.astype(np.float32, copy=False)

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
        models_dir = out_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        n_samples = int(features.shape[0])
        n_targets = int(targets.shape[1])

        if debug:
            print(f"[DEBUG] fit: n_samples={n_samples}, n_targets={n_targets}")

        targets_dense = np.asarray(targets.toarray(), dtype=np.float32)
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

            y_train = targets_dense[train_idx]
            y_valid = targets_dense[valid_idx]

            # Train PyBoost model

            alpha = float(self.config.term_neg_weight_alpha)
            neg_weight_cp = None
            if alpha > 0.0:
                w = self._term_neg_weights_from_targets(targets[train_idx], alpha=alpha)
                neg_weight_cp = cp.asarray(w, dtype=cp.float32)
                if debug:
                    w_np = w.astype(np.float32, copy=False)
                    n = int(w_np.size)
                    n_gt1 = int(np.sum(w_np > 1.0))
                    n_lt1 = int(np.sum(w_np < 1.0))
                    n_eq1 = int(n - n_gt1 - n_lt1)
                    print(
                        f"[DEBUG] neg_weight split: <1={n_lt1}/{n}, "
                        f"=1={n_eq1}/{n}, >1={n_gt1}/{n}"
                    )
                    logw = np.log(w.astype(np.float64))
                    q = np.quantile(logw, [0.01, 0.1, 0.5, 0.9, 0.99])
                    print(
                        f"[DEBUG] logw q01={q[0]:.3f} q10={q[1]:.3f} q50={q[2]:.3f} "
                        f"q90={q[3]:.3f} q99={q[4]:.3f}"
                    )

            loss = BCEWithNegWeightsLoss(neg_weight=neg_weight_cp)

            model = SketchBoost(
                loss=loss,
                ntrees=self.config.ntrees,
                lr=self.config.lr,
                es=self.config.es,
                lambda_l2=self.config.lambda_l2,
                gd_steps=self.config.gd_steps,
                subsample=self.config.subsample,
                colsample=self.config.colsample,
                min_data_in_leaf=self.config.min_data_in_leaf,
                use_hess=self.config.use_hess,
                sketch_method=self.config.sketch_method,
                sketch_outputs=self.config.sketch_outputs,
                max_bin=self.config.max_bin,
                max_depth=self.config.max_depth,
                verbose=self.config.verbose,
            )

            model.fit(
                x_train,
                y_train,
                eval_sets=[{"X": x_valid, "y": y_valid}],
            )

            print("Predicting valid set")
            probs = model.predict(x_valid)
            logits = self._probs_to_logits(probs)

            val_loss = self._bce_loss(y_valid, logits)

            # Calculate ROC AUC and Average Precision for multilabel classification (optional)
            if compute_metrics:
                print("Calculating metrics")
                val_auc, val_ap, n_valid_terms = multilabel_metrics(y_valid, probs)
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

            # Save model and normalization (compressed with gzip)
            model_path = models_dir / f"fold_{fold_id:02d}.pkl.gz"
            with gzip.open(model_path, "wb", compresslevel=6) as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save normalization parameters (method-specific)
            norm_dict = {"method": self.config.normalization}
            for key, value in norm_params.items():
                if isinstance(value, np.ndarray):
                    norm_dict[key] = value.astype(np.float32, copy=False)
                else:
                    norm_dict[key] = value
            np.savez_compressed(
                models_dir / f"fold_{fold_id:02d}_norm.npz",
                **norm_dict
            )

            # Free GPU memory after each fold
            del model, probs, logits
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

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
            "model": "pyboost",
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

    def _probs_to_logits(self, probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        probs = np.clip(probs, eps, 1.0 - eps)
        return np.log(probs / (1.0 - probs)).astype(np.float32, copy=False)

    def _bce_loss(self, y_true: np.ndarray, logits: np.ndarray) -> float:
        """Calculate the binary cross-entropy loss."""
        z = np.clip(logits, -500.0, 500.0)
        loss = np.mean(
            np.maximum(z, 0.0) - z * y_true +
            np.log1p(np.exp(-np.abs(z)))
        )
        return float(loss)

    def predict_logits_ensemble(
        self,
        features: np.ndarray,
        aspect_dir: Path,
        aspect_gt: dict[str, object] | None = None,
    ) -> np.ndarray:
        models_dir = aspect_dir / "models"
        if not models_dir.exists():
            raise FileNotFoundError(f"Missing models_dir: {models_dir}")

        fold_logits = None

        for fold_id in range(1, self.config.n_splits + 1):
            # Try compressed format first, fallback to uncompressed for backward compatibility
            model_path_compressed = models_dir / f"fold_{fold_id:02d}.pkl.gz"
            model_path_uncompressed = models_dir / f"fold_{fold_id:02d}.pkl"

            if model_path_compressed.exists():
                model_path = model_path_compressed
            elif model_path_uncompressed.exists():
                model_path = model_path_uncompressed
            else:
                raise FileNotFoundError(
                    f"Missing model file for fold {fold_id}. "
                    f"Expected {model_path_compressed} or {model_path_uncompressed}"
                )

            norm_path = models_dir / f"fold_{fold_id:02d}_norm.npz"
            if not norm_path.exists():
                raise FileNotFoundError(f"Missing fold artifacts for fold {fold_id}")

            # Load normalization parameters
            norm = np.load(norm_path)
            norm_method = str(norm.get("method", "zscore"))
            norm_params = {k: v for k, v in norm.items() if k != "method"}

            # Apply normalization using the same method as training
            x_fold = apply_normalize_test(features, norm_params, method=norm_method)

            # Load model (compressed or uncompressed)
            if model_path.suffix == ".gz":
                with gzip.open(model_path, "rb") as f:
                    model = pickle.load(f)
            else:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)

            # Predict probabilities in batches to avoid memory issues
            n_samples = x_fold.shape[0]
            batch_size = self.config.pred_batch_size

            # Calculate total number of batches
            n_batches = (n_samples + batch_size - 1) // batch_size
            first_batch_size = min(batch_size, n_samples)

            print(f"[DEBUG] Fold {fold_id}: Predicting batch 1 of {n_batches}", end='\r', flush=True)
            first_probs = model.predict(x_fold[:first_batch_size])
            n_targets = first_probs.shape[1]

            probs = np.empty((n_samples, n_targets), dtype=np.float32)
            probs[:first_batch_size] = first_probs

            batch_num = 2
            for i in range(first_batch_size, n_samples, batch_size):
                print(f"[DEBUG] Fold {fold_id}: Predicting batch {batch_num} of {n_batches}", end='\r', flush=True)
                end_idx = min(i + batch_size, n_samples)
                probs[i:end_idx] = model.predict(x_fold[i:end_idx])
                batch_num += 1

            # Convert probabilities to logits
            logits = self._probs_to_logits(probs)

            if fold_logits is None:
                fold_logits = logits
            else:
                fold_logits += logits

            # Free GPU memory after each fold prediction
            del model, probs, logits, x_fold
            gc.collect()
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass  # Ignore if pools are not initialized

        return (fold_logits / float(self.config.n_splits)).astype(np.float32, copy=False)
