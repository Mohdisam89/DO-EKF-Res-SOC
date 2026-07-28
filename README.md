# DO-EKF_Res: Data-Optimized Extended Kalman Filter with Residual BiLSTM for Synthetic Li-Ion Battery SOC Estimation

## Overview

This repository implements a hybrid battery State-of-Charge (SOC) estimation framework combining:

* **Extended Kalman Filter (EKF)** as the baseline estimator.
* **Data-Optimized Extended Kalman Filter (DO-EKF)** with optimized process and measurement covariance matrices.
* **DO-EKF_Res**, which combines DO-EKF with a Residual Bidirectional LSTM (BiLSTM) to learn and compensate the remaining estimation error.

The implementation uses a synthetic lithium-ion battery dataset with controlled model mismatch to evaluate the robustness and generalization capability of the proposed framework.

---

# Features

* Synthetic battery degradation simulation.
* Temperature-dependent first-order ECM.
* Controlled model mismatch generation.
* Adaptive covariance optimization for DO-EKF.
* Residual BiLSTM error correction.
* Confidence-gated residual correction.
* Residual gain calibration.
* Monte Carlo evaluation.
* Statistical significance analysis.

---

# Implemented Models

1. **EKF**

   * Fixed process and measurement covariance.
   * Baseline estimator.

2. **DO-EKF**

   * Learns optimal process covariance (**Q**) and measurement covariance (**R**).
   * Uses gradient-based covariance optimization.

3. **DO-EKF_Res**

   * Uses DO-EKF as the primary estimator.
   * BiLSTM predicts the remaining SOC estimation error.
   * Confidence gating suppresses unreliable residual corrections.

---

# Synthetic Dataset

The simulator generates realistic battery operating conditions including:

* Current profiles
* Voltage measurements
* Temperature variation
* State of Health (SOH)
* Capacity degradation
* Internal resistance variation
* Thermal dynamics
* Controlled model mismatch

Each Monte Carlo scenario is generated independently.

---

# Feature Set

The Residual BiLSTM uses:

* Voltage_measured
* Current_discharge
* Temperature_measured
* SOH
* SOH_delta
* time_norm
* dV_dt
* dI_dt
* discharged_Ah
* cumulative_Wh
* R_dyn_approx
* SOC_DO
* V_EKF_pred
* V_EKF_error

---

# Training Procedure

## Step 1

Generate Monte Carlo battery scenarios.

## Step 2

Train the DO-EKF covariance parameters (Q and R).

## Step 3

Estimate SOC using the optimized DO-EKF.

## Step 4

Generate residual targets

Residual = SOC_true − SOC_DO

## Step 5

Train the Residual BiLSTM using hard residual mining.

## Step 6

Calibrate the residual correction gain.

## Step 7

Evaluate all validation scenarios.

---

# Evaluation Metrics

The implementation reports only:

* RMSE
* RMSE Standard Deviation
* MAE

---

# Statistical Analysis

The implementation performs statistical comparison among the three estimators using:

* Friedman Test
* Wilcoxon Signed-Rank Test
* Holm-Bonferroni Correction
* Average Rank Analysis

These statistical tests are computed using the scenario-wise RMSE values.

---

# Output Files

The program automatically generates:

```
synthetic_3models_summary.csv
synthetic_3models_validation_metrics_long.csv
synthetic_3models_validation_results_wide.csv

stat_friedman_results.csv
stat_pairwise_wilcoxon_holm.csv
stat_average_ranks.csv

residual_training_history.csv
doekf_training_history.csv

training_progress_doekf_live.png
training_progress_residual_live.png

synthetic_3models_soc_example.png
synthetic_3models_rmse_bar.png
```

---

# Requirements

Python 3.10+

Required packages:

```
numpy
pandas
matplotlib
scipy
torch
```

Install using:

```bash
pip install numpy pandas matplotlib scipy torch
```

---

# Running the Code

```bash
python bms_doekf_residual_lstm_synthetic_model_mismatch_complete.py
```

---

# Project Structure

```
bms_doekf_residual_lstm_synthetic_model_mismatch_complete.py
README.md
results/
```

---

# Future Work
The proposed DO-EKF_Res framework is currently under preparation for journal publication. Once the manuscript is accepted and published, the corresponding citation information will be added.
