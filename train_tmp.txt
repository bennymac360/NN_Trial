import argparse
import ast
import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import wandb
from wandb.integration.keras import WandbCallback

DEFAULTS = {
    "data_path": "training_data.txt",
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "test_size": 0.2,
    "dense_sizes": [256, 128, 64],
    "seed": 42,
    "model_dir": "artifacts",
    "load_existing": None,
    "wandb_project": "nn_trial_dense_regression",
    "wandb_mode": "offline",
    "model_type": "dense",  # options: dense, cnn1d
    "cnn1d_filters": [64, 128],
    "cnn1d_kernel_sizes": [5, 3],
    "cnn1d_pool_size": 2,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_config_file(path: Path) -> dict:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_config(args: argparse.Namespace) -> SimpleNamespace:
    file_cfg = load_config_file(Path(args.config)) if args.config else {}
    cfg_dict = DEFAULTS.copy()
    cfg_dict.update({k: v for k, v in file_cfg.items() if v is not None})
    for key in DEFAULTS:
        arg_val = getattr(args, key, None)
        if arg_val is not None:
            cfg_dict[key] = arg_val
    # Normalize types
    cfg_dict["dense_sizes"] = [int(x) for x in cfg_dict["dense_sizes"]]
    cfg_dict["cnn1d_filters"] = [int(x) for x in cfg_dict["cnn1d_filters"]]
    cfg_dict["cnn1d_kernel_sizes"] = [int(x) for x in cfg_dict["cnn1d_kernel_sizes"]]
    cfg_dict["cnn1d_pool_size"] = int(cfg_dict["cnn1d_pool_size"])
    cfg_dict["epochs"] = int(cfg_dict["epochs"])
    cfg_dict["batch_size"] = int(cfg_dict["batch_size"])
    cfg_dict["seed"] = int(cfg_dict["seed"])
    cfg_dict["test_size"] = float(cfg_dict["test_size"])
    cfg_dict["learning_rate"] = float(cfg_dict["learning_rate"])
    cfg_dict["data_path"] = str(cfg_dict["data_path"])
    cfg_dict["model_type"] = str(cfg_dict["model_type"]).lower()
    return SimpleNamespace(**cfg_dict)


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load CSV with columns energy and spin (stringified list)."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path, converters={"spin": ast.literal_eval})
    features = np.vstack(df["spin"].apply(np.array)).astype(np.float32)
    targets = df["energy"].to_numpy(dtype=np.float32)
    return features, targets


def prepare_data(
    X: np.ndarray, y: np.ndarray, test_size: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    return X_train, X_val, y_train, y_val, scaler


def build_dense_model(input_dim: int, dense_sizes: List[int], learning_rate: float) -> tf.keras.Model:
    if not dense_sizes:
        raise ValueError("dense_sizes must contain at least one layer width.")
    layers = [tf.keras.layers.Input(shape=(input_dim,))]
    layers += [tf.keras.layers.Dense(units, activation="relu") for units in dense_sizes]
    layers.append(tf.keras.layers.Dense(1, activation="linear"))
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"), tf.keras.metrics.MeanSquaredError(name="mse")],
    )
    return model


def build_cnn1d_model(
    input_length: int,
    filters: List[int],
    kernel_sizes: List[int],
    pool_size: int,
    learning_rate: float,
) -> tf.keras.Model:
    if len(filters) != len(kernel_sizes):
        raise ValueError("cnn1d_filters and cnn1d_kernel_sizes must have the same length.")
    inputs = tf.keras.layers.Input(shape=(input_length, 1))
    x = inputs
    for f, k in zip(filters, kernel_sizes):
        x = tf.keras.layers.Conv1D(filters=f, kernel_size=k, activation="relu", padding="same")(x)
        if pool_size and pool_size > 1:
            x = tf.keras.layers.MaxPooling1D(pool_size=pool_size)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="linear")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"), tf.keras.metrics.MeanSquaredError(name="mse")],
    )
    return model


def build_model(cfg: SimpleNamespace, input_dim: int) -> tf.keras.Model:
    if cfg.model_type == "dense":
        return build_dense_model(input_dim=input_dim, dense_sizes=cfg.dense_sizes, learning_rate=cfg.learning_rate)
    if cfg.model_type == "cnn1d":
        return build_cnn1d_model(
            input_length=input_dim,
            filters=cfg.cnn1d_filters,
            kernel_sizes=cfg.cnn1d_kernel_sizes,
            pool_size=cfg.cnn1d_pool_size,
            learning_rate=cfg.learning_rate,
        )
    raise ValueError(f"Unsupported model_type '{cfg.model_type}'. Choose 'dense' or 'cnn1d'.")


def init_wandb(cfg: argparse.Namespace):
    if cfg.wandb_mode == "disabled":
        return wandb.init(mode="disabled")
    return wandb.init(project=cfg.wandb_project, config=vars(cfg), mode=cfg.wandb_mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dense regression model on energy dataset.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file with defaults.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to CSV dataset.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs; use 0 for eval-only.")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Adam learning rate.")
    parser.add_argument("--test-size", type=float, default=None, help="Validation split fraction.")
    parser.add_argument("--dense-sizes", type=int, nargs="+", default=None, help="Hidden layer widths.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--model-dir", type=str, default=None, help="Directory to save the trained model.")
    parser.add_argument("--load-existing", type=str, default=None, help="Optional path to a saved Keras model to continue training or eval.")
    parser.add_argument("--wandb-project", type=str, default=None, help="Weights & Biases project name.")
    parser.add_argument("--model-type", type=str, choices=["dense", "cnn1d"], default=None, help="Architecture to use.")
    parser.add_argument("--cnn1d-filters", type=int, nargs="+", default=None, help="Conv1D filter sizes per layer.")
    parser.add_argument("--cnn1d-kernel-sizes", type=int, nargs="+", default=None, help="Conv1D kernel sizes per layer (must match filters length).")
    parser.add_argument("--cnn1d-pool-size", type=int, default=None, help="MaxPool1D pool size (set to 1 or 0 to disable pooling).")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default=None,
        help="wandb logging mode. Use offline when internet is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = resolve_config(args)
    set_seed(cfg.seed)

    data_path = Path(cfg.data_path)
    X, y = load_dataset(data_path)
    X_train, X_val, y_train, y_val, scaler = prepare_data(X, y, test_size=cfg.test_size, seed=cfg.seed)
    input_dim = X_train.shape[1]

    if cfg.model_type == "cnn1d":
        X_train = X_train.reshape(-1, input_dim, 1)
        X_val = X_val.reshape(-1, input_dim, 1)

    if cfg.load_existing:
        model = tf.keras.models.load_model(cfg.load_existing)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
            loss="mse",
            metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"), tf.keras.metrics.MeanSquaredError(name="mse")],
        )
    else:
        model = build_model(cfg=cfg, input_dim=input_dim)

    run = init_wandb(cfg)
    callbacks = []
    if cfg.wandb_mode != "disabled":
        callbacks.append(WandbCallback(save_model=False, save_graph=False))

    if cfg.epochs > 0:
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            shuffle=True,
            callbacks=callbacks,
            verbose=1,
        )

    eval_metrics = model.evaluate(X_val, y_val, verbose=0, return_dict=True)
    if run is not None:
        wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()})
        run.summary.update({f"eval/{k}": v for k, v in eval_metrics.items()})
        run.finish()

    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if cfg.epochs > 0:
        model_path = model_dir / "model.keras"
        model.save(model_path)
        print(f"Model saved to {model_path}")
    print(f"Validation metrics: {eval_metrics}")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF INFO logs.
    main()
