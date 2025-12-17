# Dense Regression RNN (Keras)

Train a fully connected, dense-only neural network in Keras/TensorFlow to predict `thickness` from 109-length numeric sequences stored in `training_data.txt`. We log metrics to Weights & Biases (wandb) and save a Keras model artifact.

## Setup
- Ensure Python 3.10+ is available.
- Create & activate a virtual environment (PowerShell):
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```
- W&B: either set `WANDB_API_KEY` env var or run `wandb login`. If you are offline, use `--wandb-mode offline` when running the script.

## Data
- Training file: `training_data.txt` (CSV with columns `thickness` and `values`, where `values` is a stringified list of 109 numbers).
- The script automatically parses and standardizes features; no manual preprocessing needed.

## Train
Edit `config.yaml` to set all defaults (data path, epochs, batch size, dense layer widths, wandb mode, etc.). CLI flags override anything in the config.

```powershell
python train.py --config config.yaml
```
### Architecture selection
- Set `model_type: dense` (default) to use the fully-connected stack defined by `dense_sizes`.
- Set `model_type: cnn1d` to use a 1D CNN; configure `cnn1d_filters`, `cnn1d_kernel_sizes`, and `cnn1d_pool_size` in `config.yaml`.
- Learning-rate reduction: control with `lr_reduce_on_plateau`, `lr_patience`, `lr_factor`, `lr_min` in `config.yaml` (ReduceLROnPlateau on `val_loss`).

CLI overrides (examples):
- `--epochs 10` : override epochs from config.
- `--dense-sizes 256 128 64 32` : create 4 hidden Dense layers with those widths.
- `--model-type cnn1d --cnn1d-filters 64 128 256 --cnn1d-kernel-sizes 7 5 3 --cnn1d-pool-size 2` : run a 3-layer 1D CNN with custom kernels and pooling.
- `--wandb-mode offline` : set wandb mode for this run.

## Evaluate
Validation metrics are reported at the end of training and logged to wandb (if enabled). To run an evaluation-only pass on the saved model:
```powershell
python train.py --data-path training_data.txt --epochs 0 --load-existing artifacts/model.keras
```
This will load the saved model, compute metrics on the held-out validation split, and exit.

## Notes
- Loss: Mean Squared Error; optimizer: Adam; activations: ReLU on hidden layers, linear on the output for regression.
- The number of hidden Dense layers is determined by `dense_sizes` in `config.yaml` (or CLI); provide any length list to build that many layers.
- For `cnn1d`, the feature vector is reshaped to `(sequence_length, 1)`, then a stack of Conv1D + optional MaxPool layers (from `cnn1d_filters`/`cnn1d_kernel_sizes`/`cnn1d_pool_size`) feeds into a Dense head for regression.
- ReduceLROnPlateau is enabled by default; tune or disable it via config.
- Deterministic seeds are set for reproducibility; disable GPU nondeterminism separately if needed.
