# HybridGAN-BWE MLOps & MLflow Experiment Tracking Guide

This module provides complete, non-breaking **MLflow Experiment Tracking and MLOps Infrastructure** for the **HybridGAN Speech Bandwidth Extension** project.

---

## 🚀 Key Features

* **Automatic Parameter Logging**: Logs hyperparameters (`configs/config.yaml`), random seed, batch size, learning rates, epochs, segment length, optimizer (`AdamW`), and scheduler (`ExponentialLR`).
* **Environment & Library Version Tracking**: Logs exact versions of `torch`, `torchaudio`, `librosa`, `pesq`, `pystoi`, `gradio`, `scipy`, `numpy`, `python`, and `mlflow`.
* **Per-Epoch Metric Curves**: Tracks generator loss (`loss_g`), discriminator loss (`loss_d`), STFT loss, phase loss, Mel loss, feature matching loss, and validation loss metrics (`val_loss`, `val_stft`, `val_phase`, `val_mel`) across training epochs.
* **Artifact Management**: Logs model checkpoints (`best_model.pth`, `checkpoint_epoch_X.pth`), configuration files (`config.yaml`), Markdown evaluation reports, and spectrogram plots directly to MLflow artifact storage.
* **Test Evaluation Tracking**: Logs final quality metrics (PESQ, STOI, SI-SDR, LSD, RTF, GPU Latency).
* **Reproducibility**: Enforces deterministic random seeds across PyTorch, CUDA, NumPy, and Python `random`.

---

## 📦 1. Installation

Ensure your virtual environment is active, then install `mlflow`:

```powershell
.\.venv\Scripts\pip install mlflow
```

---

## 🖥️ 2. Starting the MLflow UI

To view the interactive MLflow dashboard in your web browser:

### Option A: Using the MLOps Helper Script (Recommended)
```powershell
.\.venv\Scripts\python -m mlops.start_ui --port 5000
```

### Option B: Using MLflow CLI Directly
```powershell
.\.venv\Scripts\mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5000
```

Open your browser and navigate to: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🏋️ 3. How to Train with MLflow Tracking

To run model training with automatic MLflow metric and artifact tracking:

### Run Training via MLOps Module:
```powershell
.\.venv\Scripts\python -m mlops.train --config mlops/config.yaml --experiment_name "HybridGAN-BWE" --seed 42
```

### Options:
* `--config`: Path to configuration YAML file (default: `configs/config.yaml`).
* `--experiment_name`: Name of the MLflow experiment (default: `HybridGAN-BWE`).
* `--run_name`: Custom name for the run in MLflow UI.
* `--seed`: Random seed for reproducibility (default: `42`).

---

## 📊 4. How to Evaluate & Track Test Metrics

To evaluate a trained checkpoint on the test set and log PESQ, STOI, SI-SDR, LSD, and RTF to MLflow:

```powershell
.\.venv\Scripts\python -m mlops.evaluate_mlflow --checkpoint checkpoints/lightweight/best_model.pth --num_samples 200
```

This will:
1. Compute average metrics on 200 test samples.
2. Save a formatted Markdown report to `outputs/mlflow_evaluation_report.md`.
3. Log metrics (`test_pesq_wb`, `test_stoi`, `test_lsd_db`, `test_sisdr_db`, `test_rtf`, `test_latency_ms`) to MLflow.
4. Upload the generated report and checkpoint as MLflow artifacts.

---

## 🔍 5. How to Compare Runs in the MLflow UI

1. Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.
2. Select the experiment **`HybridGAN-BWE`** from the left panel.
3. Select multiple runs using the checkboxes.
4. Click **Compare** to view overlay metric graphs (e.g. `train_loss_g`, `val_loss`, `lr_g`) and side-by-side hyperparameter tables.
5. Click on any individual run to browse uploaded model checkpoints (`best_model.pth`) and evaluation artifacts under the **Artifacts** tab.
