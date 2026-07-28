# nasa_data_audit_before_training.py
# ============================================================
# STEP 1 ONLY: NASA Battery Dataset Audit Before Any Training
# Checks data reading, current sign, time validity, and SOC(t)
# computed by Coulomb counting from current/time.
# ============================================================

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# NASA Dataset
DATASET_DIR = Path("NASA_Dataset")

METADATA_PATH = DATASET_DIR / "metadata.csv"
DATA_DIR = DATASET_DIR / "data"

BATTERY_IDS = ["B0005", "B0006", "B0007", "B0018"]

# اجعلها 20 للفحص السريع، ثم None لفحص كل الدورات.
MAX_CYCLES_PER_BATTERY = 20

OUT_DIR = "nasa_data_audit_results"
os.makedirs(OUT_DIR, exist_ok=True)


def find_cycle_file(data_dir, filename):
    data_dir = Path(data_dir)
    f = str(filename).strip()

    candidates = [
        data_dir / f,
        data_dir / f"{f}.csv",
        data_dir / f"{f}.xlsx",
        data_dir / f"{f}.xls",
    ]

    for c in candidates:
        if c.exists():
            return c

    matches = list(data_dir.rglob(f))
    if matches:
        return matches[0]

    matches = list(data_dir.rglob(f"{f}.*"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Cannot find data file for filename={filename}")


def read_cycle_file(path):
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_excel(path)

    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_required_columns(df):
    required = {
        "Voltage_measured": None,
        "Current_measured": None,
        "Temperature_measured": None,
        "Time": None,
    }

    lower_map = {c.lower(): c for c in df.columns}

    for key in required:
        if key in df.columns:
            required[key] = key
        elif key.lower() in lower_map:
            required[key] = lower_map[key.lower()]
        else:
            raise KeyError(
                f"Missing required column: {key}. Available columns: {list(df.columns)}"
            )

    return required


def prepare_single_discharge_cycle(df, capacity_ah):
    cols = get_required_columns(df)

    V = pd.to_numeric(df[cols["Voltage_measured"]], errors="coerce").to_numpy(dtype=float)
    I_raw = pd.to_numeric(df[cols["Current_measured"]], errors="coerce").to_numpy(dtype=float)
    T = pd.to_numeric(df[cols["Temperature_measured"]], errors="coerce").to_numpy(dtype=float)
    time = pd.to_numeric(df[cols["Time"]], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(V) & np.isfinite(I_raw) & np.isfinite(T) & np.isfinite(time)
    V, I_raw, T, time = V[mask], I_raw[mask], T[mask], time[mask]

    if len(V) < 20:
        raise ValueError("Too few valid samples.")

    order = np.argsort(time)
    V, I_raw, T, time = V[order], I_raw[order], T[order], time[order]

    _, unique_idx = np.unique(time, return_index=True)
    V, I_raw, T, time = V[unique_idx], I_raw[unique_idx], T[unique_idx], time[unique_idx]

    # NASA discharge current may be negative. Convert to positive discharge convention.
    if np.nanmean(I_raw) < 0:
        I_dis = -I_raw
        current_sign_convention = "negative_raw_converted_to_positive_discharge"
    else:
        I_dis = I_raw.copy()
        current_sign_convention = "positive_raw_used_as_discharge"

    I_dis = np.maximum(I_dis, 0.0)

    dt = np.diff(time, prepend=time[0])
    if len(dt) > 1:
        dt[0] = np.median(dt[1:])
    dt = np.clip(dt, 0.0, 60.0)

    discharged_ah = np.cumsum(I_dis * dt) / 3600.0

    if not np.isfinite(capacity_ah) or capacity_ah <= 0:
        capacity_ah = float(discharged_ah[-1])

    # Correct SOC reference:
    # SOC(t) = 1 - discharged_Ah(t) / cycle_capacity_Ah
    SOC = 1.0 - discharged_ah / capacity_ah
    SOC = np.clip(SOC, 0.0, 1.0)

    return {
        "time": time.astype(float),
        "V": V.astype(float),
        "I_raw": I_raw.astype(float),
        "I_dis": I_dis.astype(float),
        "T": T.astype(float),
        "dt": dt.astype(float),
        "discharged_ah": discharged_ah.astype(float),
        "SOC": SOC.astype(float),
        "current_sign_convention": current_sign_convention,
    }


def monotonicity_score_decreasing(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.nan
    return float(np.mean(np.diff(x) <= 1e-9))


def monotonicity_score_increasing(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.nan
    return float(np.mean(np.diff(x) >= -1e-9))


def audit_cycle(cycle, battery_id, filename, capacity_ah, q_nom):
    time = cycle["time"]
    V = cycle["V"]
    I_raw = cycle["I_raw"]
    I_dis = cycle["I_dis"]
    T = cycle["T"]
    dt = cycle["dt"]
    discharged_ah = cycle["discharged_ah"]
    SOC = cycle["SOC"]

    soh = capacity_ah / q_nom if q_nom > 0 else np.nan

    discharged_end = float(discharged_ah[-1])
    cap_abs_error = float(abs(discharged_end - capacity_ah))
    cap_rel_error = float(cap_abs_error / max(capacity_ah, 1e-12))

    soc_start = float(SOC[0])
    soc_end = float(SOC[-1])
    soc_drop = float(soc_start - soc_end)

    flag_time_increasing = bool(np.all(np.diff(time) > 0))
    flag_soc_decreasing = bool(monotonicity_score_decreasing(SOC) > 0.98)
    flag_voltage_decreasing_mostly = bool(monotonicity_score_decreasing(V) > 0.60)
    flag_capacity_match = bool(cap_rel_error < 0.15)
    flag_soc_range = bool((SOC.min() >= -1e-6) and (SOC.max() <= 1.0 + 1e-6))
    flag_soc_start_ok = bool(soc_start > 0.95)
    flag_soc_drop_ok = bool(soc_drop > 0.50)

    audit_status = "PASS"
    issues = []

    if not flag_time_increasing:
        audit_status = "FAIL"
        issues.append("time_not_strictly_increasing")

    if not flag_soc_range:
        audit_status = "FAIL"
        issues.append("soc_out_of_range")

    if not flag_soc_decreasing:
        audit_status = "WARN"
        issues.append("soc_not_monotonic_decreasing")

    if not flag_capacity_match:
        audit_status = "WARN"
        issues.append("discharged_Ah_not_close_to_metadata_capacity")

    if not flag_soc_start_ok:
        audit_status = "WARN"
        issues.append("soc_start_not_near_one")

    if not flag_soc_drop_ok:
        audit_status = "WARN"
        issues.append("soc_drop_too_small")

    return {
        "battery_id": battery_id,
        "filename": filename,
        "samples": len(time),
        "capacity_metadata_Ah": float(capacity_ah),
        "Q_nom_Ah": float(q_nom),
        "SOH": float(soh),
        "time_start_s": float(time[0]),
        "time_end_s": float(time[-1]),
        "dt_median_s": float(np.median(dt[1:])) if len(dt) > 1 else np.nan,
        "dt_min_s": float(np.min(dt)),
        "dt_max_s": float(np.max(dt)),
        "V_start": float(V[0]),
        "V_end": float(V[-1]),
        "V_min": float(np.min(V)),
        "V_max": float(np.max(V)),
        "I_raw_mean": float(np.mean(I_raw)),
        "I_dis_mean": float(np.mean(I_dis)),
        "T_mean": float(np.mean(T)),
        "T_min": float(np.min(T)),
        "T_max": float(np.max(T)),
        "discharged_Ah_end": discharged_end,
        "capacity_abs_error_Ah": cap_abs_error,
        "capacity_rel_error": cap_rel_error,
        "SOC_start": soc_start,
        "SOC_end": soc_end,
        "SOC_drop": soc_drop,
        "SOC_min": float(np.min(SOC)),
        "SOC_max": float(np.max(SOC)),
        "SOC_decreasing_score": monotonicity_score_decreasing(SOC),
        "V_decreasing_score": monotonicity_score_decreasing(V),
        "Ah_increasing_score": monotonicity_score_increasing(discharged_ah),
        "current_sign_convention": cycle["current_sign_convention"],
        "flag_time_increasing": flag_time_increasing,
        "flag_soc_decreasing": flag_soc_decreasing,
        "flag_voltage_decreasing_mostly": flag_voltage_decreasing_mostly,
        "flag_capacity_match": flag_capacity_match,
        "flag_soc_range": flag_soc_range,
        "flag_soc_start_ok": flag_soc_start_ok,
        "flag_soc_drop_ok": flag_soc_drop_ok,
        "audit_status": audit_status,
        "issues": ";".join(issues),
    }


def plot_cycle_audit(cycle, row, out_dir):
    time_h = (cycle["time"] - cycle["time"][0]) / 3600.0

    fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

    ax[0].plot(time_h, cycle["V"], lw=1.2)
    ax[0].set_ylabel("Voltage (V)")
    ax[0].grid(alpha=0.3)

    ax[1].plot(time_h, cycle["I_raw"], lw=1.0, label="Raw current")
    ax[1].plot(time_h, cycle["I_dis"], "--", lw=1.0, label="Discharge current")
    ax[1].set_ylabel("Current (A)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    ax[2].plot(time_h, cycle["T"], lw=1.0)
    ax[2].set_ylabel("Temperature (°C)")
    ax[2].grid(alpha=0.3)

    ax[3].plot(time_h, cycle["SOC"]*100, lw=1.5)
    ax[3].set_ylabel("SOC (%)")
    ax[3].set_xlabel("Time (h)")
    ax[3].grid(alpha=0.3)

    title = (
        f"{row['battery_id']} | {row['filename']} | "
        f"SOH={row['SOH']:.3f} | Status={row['audit_status']}"
    )
    fig.suptitle(title, fontsize=11, fontweight="bold")

    plt.tight_layout()
    safe_name = f"audit_example_{row['battery_id']}_{str(row['filename']).replace('.', '_')}"
    plt.savefig(os.path.join(out_dir, safe_name + ".png"), dpi=250, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, safe_name + ".pdf"), dpi=250, bbox_inches="tight")
    plt.close()


def run_audit():
    print("\n" + "█"*88)
    print(" NASA Battery Data Audit Before Training")
    print("█"*88)

    print(f"Metadata: {METADATA_PATH}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Batteries: {BATTERY_IDS}")
    print(f"MAX_CYCLES_PER_BATTERY: {MAX_CYCLES_PER_BATTERY}")

    meta = pd.read_csv(METADATA_PATH)
    meta.columns = [str(c).strip() for c in meta.columns]

    required_meta = ["type", "battery_id", "filename", "Capacity"]
    for c in required_meta:
        if c not in meta.columns:
            raise KeyError(f"metadata.csv missing column: {c}")

    dis = meta[meta["type"].astype(str).str.lower().eq("discharge")].copy()
    dis["Capacity"] = pd.to_numeric(dis["Capacity"], errors="coerce")
    dis = dis[np.isfinite(dis["Capacity"]) & (dis["Capacity"] > 0)].copy()
    dis = dis[dis["battery_id"].astype(str).isin(BATTERY_IDS)].copy()

    if "test_id" in dis.columns:
        dis["test_id_num"] = pd.to_numeric(dis["test_id"], errors="coerce")
        dis = dis.sort_values(["battery_id", "test_id_num"])
    else:
        dis = dis.sort_values(["battery_id"])

    q_nom_by_batt = dis.groupby("battery_id")["Capacity"].max().to_dict()

    all_rows = []
    first_cycle_by_batt = {}

    for batt, g in dis.groupby("battery_id"):
        batt = str(batt)

        if MAX_CYCLES_PER_BATTERY is not None:
            g = g.head(MAX_CYCLES_PER_BATTERY)

        q_nom = float(q_nom_by_batt[batt])

        print(f"\nBattery {batt}: Q_nom={q_nom:.6f} Ah | cycles to audit={len(g)}")

        for _, row in g.iterrows():
            fname = str(row["filename"])
            cap = float(row["Capacity"])

            try:
                fpath = find_cycle_file(DATA_DIR, fname)
                df = read_cycle_file(fpath)
                cycle = prepare_single_discharge_cycle(df, capacity_ah=cap)
                audit_row = audit_cycle(cycle, batt, fname, cap, q_nom)
                all_rows.append(audit_row)

                if batt not in first_cycle_by_batt:
                    first_cycle_by_batt[batt] = (cycle, audit_row)

                print(
                    f"  {fname:<12} | {audit_row['audit_status']:<4} | "
                    f"SOH={audit_row['SOH']:.3f} | "
                    f"SOC {audit_row['SOC_start']:.3f}->{audit_row['SOC_end']:.3f} | "
                    f"Ah_end={audit_row['discharged_Ah_end']:.3f} | "
                    f"Cap={cap:.3f} | "
                    f"CapErr={audit_row['capacity_rel_error']*100:.1f}%"
                )

            except Exception as e:
                fail_row = {
                    "battery_id": batt,
                    "filename": fname,
                    "audit_status": "FAIL",
                    "issues": str(e),
                }
                all_rows.append(fail_row)
                print(f"  {fname:<12} | FAIL | {e}")

    audit_df = pd.DataFrame(all_rows)
    audit_csv = os.path.join(OUT_DIR, "nasa_cycle_audit_summary.csv")
    audit_df.to_csv(audit_csv, index=False)

    summary_rows = []
    for batt, g in audit_df.groupby("battery_id"):
        summary_rows.append({
            "battery_id": batt,
            "n_cycles": len(g),
            "n_pass": int((g["audit_status"] == "PASS").sum()) if "audit_status" in g else 0,
            "n_warn": int((g["audit_status"] == "WARN").sum()) if "audit_status" in g else 0,
            "n_fail": int((g["audit_status"] == "FAIL").sum()) if "audit_status" in g else 0,
            "SOH_min": pd.to_numeric(g.get("SOH"), errors="coerce").min(),
            "SOH_max": pd.to_numeric(g.get("SOH"), errors="coerce").max(),
            "SOC_end_mean": pd.to_numeric(g.get("SOC_end"), errors="coerce").mean(),
            "capacity_rel_error_mean": pd.to_numeric(g.get("capacity_rel_error"), errors="coerce").mean(),
            "V_start_mean": pd.to_numeric(g.get("V_start"), errors="coerce").mean(),
            "V_end_mean": pd.to_numeric(g.get("V_end"), errors="coerce").mean(),
        })

    battery_summary_df = pd.DataFrame(summary_rows)
    battery_summary_csv = os.path.join(OUT_DIR, "nasa_battery_audit_summary.csv")
    battery_summary_df.to_csv(battery_summary_csv, index=False)

    for _, (cycle, row) in first_cycle_by_batt.items():
        plot_cycle_audit(cycle, row, OUT_DIR)

    print("\n" + "═"*88)
    print("AUDIT COMPLETE")
    print("═"*88)
    print(f"Cycle audit CSV  : {audit_csv}")
    print(f"Battery audit CSV: {battery_summary_csv}")
    print(f"Plots saved in   : {OUT_DIR}")

    print("\nBattery-level summary:")
    print(battery_summary_df.to_string(index=False))

    print("\nImportant checks before training:")
    print("  1) SOC_start should be close to 1.")
    print("  2) SOC_end should be near 0 or much smaller than SOC_start.")
    print("  3) discharged_Ah_end should be close to metadata Capacity.")
    print("  4) SOC should monotonically decrease.")
    print("  5) Voltage should mostly decrease during discharge.")

    return audit_df, battery_summary_df


if __name__ == "__main__":
    audit_df, battery_summary_df = run_audit()


# nasa_professional_data_preparation_before_training.py
# ============================================================
# Professional NASA Battery Dataset Preparation BEFORE Training
#
# Purpose:
#   Convert NASA raw discharge files into a clean, validated, reproducible
#   machine-learning / filtering-ready dataset.
#
# This script performs:
#   1) Read NASA metadata.csv and discharge files.
#   2) Use only selected batteries: B0005, B0006, B0007, B0018.
#   3) Correctly compute SOC(t) using Coulomb counting:
#
#          SOC(t) = 1 - discharged_Ah(t) / Capacity_cycle
#
#   4) Compute SOH from cycle capacity:
#
#          SOH = Capacity_cycle / Q_nom_battery
#
#   5) Extract engineered features:
#          I, V, T, SOC, SOH, time, dV/dt, dI/dt,
#          cumulative Ah, cumulative Wh, normalized time,
#          cycle index, capacity, Q_nom, approximate dynamic resistance.
#
#   6) Audit each cycle:
#          time validity, current sign, capacity consistency,
#          SOC monotonicity, voltage trend, outlier flags.
#
#   7) Save:
#          clean per-sample dataset CSV
#          cycle-level summary CSV
#          battery-level summary CSV
#          train/test fold definition CSV for leave-one-battery-out
#          diagnostic plots
#
# Run:
#   python nasa_professional_data_preparation_before_training.py
#
# Requirements:
#   pip install numpy pandas matplotlib openpyxl
# ============================================================

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# 0. User configuration
# ============================================================

METADATA_PATH = r"C:\Users\engmo\Desktop\BMS online reseach\Dataset\NASA\metadata.csv"
DATA_DIR      = r"C:\Users\engmo\Desktop\BMS online reseach\Dataset\NASA\data"

BATTERY_IDS = ["B0005", "B0006", "B0007", "B0018"]

# None = all discharge cycles.
# For fast testing, set to e.g. 20.
MAX_CYCLES_PER_BATTERY = None

# Resampling is optional.
# If RESAMPLE_N=None, original sample lengths are preserved.
# For equal-length deep learning sequences, set RESAMPLE_N=700 or 1200.
RESAMPLE_N = None

OUT_DIR = "nasa_prepared_clean_dataset"
PLOTS_DIR = os.path.join(OUT_DIR, "diagnostic_plots")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Filtering rules.
MIN_SAMPLES = 20
MAX_DT_ALLOWED = 60.0
CAPACITY_REL_ERROR_WARN = 0.05
CAPACITY_REL_ERROR_FAIL = 0.15

# Robust outlier limits for sanity checking.
VOLTAGE_VALID_RANGE = (2.0, 4.5)
TEMP_VALID_RANGE = (-20.0, 80.0)
CURRENT_VALID_RANGE = (-20.0, 20.0)

SAVE_PER_BATTERY_FILES = True
SAVE_FULL_DATASET_CSV = True
SAVE_PARQUET_IF_AVAILABLE = True

RANDOM_SEED = 42


# ============================================================
# 1. General utilities
# ============================================================

def safe_float(x, default=np.nan):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default


def find_cycle_file(data_dir: str, filename: str) -> Path:
    data_dir = Path(data_dir)
    f = str(filename).strip()

    candidates = [
        data_dir / f,
        data_dir / f"{f}.csv",
        data_dir / f"{f}.xlsx",
        data_dir / f"{f}.xls",
    ]

    for c in candidates:
        if c.exists():
            return c

    matches = list(data_dir.rglob(f))
    if matches:
        return matches[0]

    matches = list(data_dir.rglob(f"{f}.*"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Cannot find data file for filename={filename}")


def read_cycle_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_excel(path)

    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_required_columns(df: pd.DataFrame):
    required = {
        "Voltage_measured": None,
        "Current_measured": None,
        "Temperature_measured": None,
        "Time": None,
    }

    lower_map = {c.lower(): c for c in df.columns}

    for key in required:
        if key in df.columns:
            required[key] = key
        elif key.lower() in lower_map:
            required[key] = lower_map[key.lower()]
        else:
            raise KeyError(
                f"Missing required column: {key}. Available columns: {list(df.columns)}"
            )

    return required


def monotonicity_score_decreasing(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.nan
    return float(np.mean(np.diff(x) <= 1e-9))


def monotonicity_score_increasing(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.nan
    return float(np.mean(np.diff(x) >= -1e-9))


# ============================================================
# 2. NASA metadata
# ============================================================

def load_metadata(metadata_path: str,
                  battery_ids=None,
                  max_cycles_per_battery=None) -> pd.DataFrame:
    meta = pd.read_csv(metadata_path)
    meta.columns = [str(c).strip() for c in meta.columns]

    required_meta = ["type", "battery_id", "filename", "Capacity"]
    for c in required_meta:
        if c not in meta.columns:
            raise KeyError(f"metadata.csv missing required column: {c}")

    dis = meta[meta["type"].astype(str).str.lower().eq("discharge")].copy()
    dis["battery_id"] = dis["battery_id"].astype(str)
    dis["filename"] = dis["filename"].astype(str)
    dis["Capacity"] = pd.to_numeric(dis["Capacity"], errors="coerce")
    dis = dis[np.isfinite(dis["Capacity"]) & (dis["Capacity"] > 0)].copy()

    if battery_ids is not None:
        dis = dis[dis["battery_id"].isin(battery_ids)].copy()

    if "test_id" in dis.columns:
        dis["test_id_num"] = pd.to_numeric(dis["test_id"], errors="coerce")
        dis = dis.sort_values(["battery_id", "test_id_num", "filename"])
    else:
        dis = dis.sort_values(["battery_id", "filename"])

    if max_cycles_per_battery is not None:
        dis = dis.groupby("battery_id", group_keys=False).head(max_cycles_per_battery)

    q_nom_by_batt = dis.groupby("battery_id")["Capacity"].max().to_dict()
    dis["Q_nom"] = dis["battery_id"].map(q_nom_by_batt)
    dis["SOH"] = dis["Capacity"] / dis["Q_nom"]

    return dis.reset_index(drop=True)


# ============================================================
# 3. Per-cycle preparation
# ============================================================

def parse_raw_cycle(df: pd.DataFrame):
    cols = get_required_columns(df)

    V = pd.to_numeric(df[cols["Voltage_measured"]], errors="coerce").to_numpy(dtype=float)
    I_raw = pd.to_numeric(df[cols["Current_measured"]], errors="coerce").to_numpy(dtype=float)
    T = pd.to_numeric(df[cols["Temperature_measured"]], errors="coerce").to_numpy(dtype=float)
    time = pd.to_numeric(df[cols["Time"]], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(V) & np.isfinite(I_raw) & np.isfinite(T) & np.isfinite(time)
    V, I_raw, T, time = V[mask], I_raw[mask], T[mask], time[mask]

    if len(V) < MIN_SAMPLES:
        raise ValueError(f"Too few valid samples: {len(V)}")

    # Sort by time and remove duplicated timestamps.
    order = np.argsort(time)
    V, I_raw, T, time = V[order], I_raw[order], T[order], time[order]

    _, unique_idx = np.unique(time, return_index=True)
    V, I_raw, T, time = V[unique_idx], I_raw[unique_idx], T[unique_idx], time[unique_idx]

    if len(V) < MIN_SAMPLES:
        raise ValueError(f"Too few samples after duplicate-time removal: {len(V)}")

    return time, V, I_raw, T


def compute_discharge_current(I_raw):
    # Positive-discharge convention.
    if np.nanmean(I_raw) < 0:
        I_dis = -I_raw
        convention = "negative_raw_converted_to_positive_discharge"
    else:
        I_dis = I_raw.copy()
        convention = "positive_raw_used_as_discharge"

    # NASA discharge should be non-negative after conversion.
    I_dis = np.maximum(I_dis, 0.0)
    return I_dis, convention


def compute_dt(time):
    dt = np.diff(time, prepend=time[0])
    if len(dt) > 1:
        dt[0] = np.median(dt[1:])
    dt = np.clip(dt, 0.0, MAX_DT_ALLOWED)
    return dt


def resample_cycle(time, V, I_raw, I_dis, T, n_points):
    if n_points is None:
        return time, V, I_raw, I_dis, T

    t0, t1 = float(time[0]), float(time[-1])
    if t1 <= t0:
        raise ValueError("Invalid time vector; final time <= initial time.")

    t_new = np.linspace(t0, t1, n_points)

    Vn = np.interp(t_new, time, V)
    Iraw_n = np.interp(t_new, time, I_raw)
    Idis_n = np.interp(t_new, time, I_dis)
    Tn = np.interp(t_new, time, T)

    return t_new, Vn, Iraw_n, Idis_n, Tn


def prepare_cycle_dataframe(raw_df: pd.DataFrame,
                            battery_id: str,
                            filename: str,
                            cycle_index: int,
                            capacity_ah: float,
                            q_nom_ah: float,
                            soh: float,
                            resample_n=None):
    time, V, I_raw, T = parse_raw_cycle(raw_df)
    I_dis, current_convention = compute_discharge_current(I_raw)

    time, V, I_raw, I_dis, T = resample_cycle(time, V, I_raw, I_dis, T, resample_n)

    # Re-zero time after optional resampling.
    time = time - time[0]
    dt = compute_dt(time)

    discharged_ah = np.cumsum(I_dis * dt) / 3600.0
    cumulative_wh = np.cumsum(V * I_dis * dt) / 3600.0

    # Correct SOC ground-truth reference.
    SOC = 1.0 - discharged_ah / max(capacity_ah, 1e-12)
    SOC = np.clip(SOC, 0.0, 1.0)

    # Features.
    dV_dt = np.gradient(V, time + 1e-12)
    dI_dt = np.gradient(I_dis, time + 1e-12)

    # Approximate dynamic resistance:
    # R_dyn = -dV/dI, computed only where dI is not tiny.
    dV = np.gradient(V)
    dI = np.gradient(I_dis)
    R_dyn = np.full_like(V, np.nan, dtype=float)
    valid_di = np.abs(dI) > 1e-4
    R_dyn[valid_di] = -dV[valid_di] / dI[valid_di]
    R_dyn = np.clip(R_dyn, -5.0, 5.0)

    # Fill R_dyn NaNs by median, then zero if all NaN.
    if np.any(np.isfinite(R_dyn)):
        med_r = np.nanmedian(R_dyn)
        R_dyn = np.where(np.isfinite(R_dyn), R_dyn, med_r)
    else:
        R_dyn = np.zeros_like(V)

    time_norm = time / max(time[-1], 1e-12)
    sample_index = np.arange(len(time), dtype=int)

    out = pd.DataFrame({
        "battery_id": battery_id,
        "filename": filename,
        "cycle_index": cycle_index,
        "sample_index": sample_index,

        "time_s": time.astype(np.float32),
        "time_norm": time_norm.astype(np.float32),
        "dt_s": dt.astype(np.float32),

        "Voltage_measured": V.astype(np.float32),
        "Current_raw": I_raw.astype(np.float32),
        "Current_discharge": I_dis.astype(np.float32),
        "Temperature_measured": T.astype(np.float32),

        "SOC": SOC.astype(np.float32),
        "SOH": np.full(len(time), soh, dtype=np.float32),
        "SOH_delta": np.full(len(time), 1.0-soh, dtype=np.float32),

        "Capacity_Ah": np.full(len(time), capacity_ah, dtype=np.float32),
        "Q_nom_Ah": np.full(len(time), q_nom_ah, dtype=np.float32),

        "discharged_Ah": discharged_ah.astype(np.float32),
        "cumulative_Wh": cumulative_wh.astype(np.float32),

        "dV_dt": dV_dt.astype(np.float32),
        "dI_dt": dI_dt.astype(np.float32),
        "R_dyn_approx": R_dyn.astype(np.float32),
    })

    meta = {
        "current_convention": current_convention,
        "n_samples": len(out),
        "time_end_s": float(time[-1]),
        "capacity_metadata_Ah": float(capacity_ah),
        "discharged_Ah_end": float(discharged_ah[-1]),
        "cumulative_Wh_end": float(cumulative_wh[-1]),
        "capacity_abs_error_Ah": float(abs(discharged_ah[-1] - capacity_ah)),
        "capacity_rel_error": float(abs(discharged_ah[-1] - capacity_ah) / max(capacity_ah, 1e-12)),
        "SOC_start": float(SOC[0]),
        "SOC_end": float(SOC[-1]),
        "SOC_drop": float(SOC[0] - SOC[-1]),
        "V_start": float(V[0]),
        "V_end": float(V[-1]),
        "V_min": float(np.min(V)),
        "V_max": float(np.max(V)),
        "I_raw_mean": float(np.mean(I_raw)),
        "I_dis_mean": float(np.mean(I_dis)),
        "T_mean": float(np.mean(T)),
        "T_min": float(np.min(T)),
        "T_max": float(np.max(T)),
        "dt_median_s": float(np.median(dt[1:])) if len(dt) > 1 else np.nan,
        "dt_min_s": float(np.min(dt)),
        "dt_max_s": float(np.max(dt)),
        "SOC_decreasing_score": monotonicity_score_decreasing(SOC),
        "V_decreasing_score": monotonicity_score_decreasing(V),
        "Ah_increasing_score": monotonicity_score_increasing(discharged_ah),
    }

    return out, meta


# ============================================================
# 4. Audit and outlier detection
# ============================================================

def classify_cycle_status(meta):
    issues = []
    status = "PASS"

    if meta["n_samples"] < MIN_SAMPLES:
        status = "FAIL"
        issues.append("too_few_samples")

    if meta["capacity_rel_error"] > CAPACITY_REL_ERROR_FAIL:
        status = "FAIL"
        issues.append("capacity_error_exceeds_fail_threshold")
    elif meta["capacity_rel_error"] > CAPACITY_REL_ERROR_WARN:
        status = "WARN"
        issues.append("capacity_error_exceeds_warning_threshold")

    if not (0.95 <= meta["SOC_start"] <= 1.001):
        status = "WARN" if status != "FAIL" else status
        issues.append("soc_start_not_near_one")

    if meta["SOC_drop"] < 0.50:
        status = "WARN" if status != "FAIL" else status
        issues.append("soc_drop_too_small")

    if meta["SOC_decreasing_score"] < 0.98:
        status = "WARN" if status != "FAIL" else status
        issues.append("soc_not_monotonic_decreasing")

    if not (VOLTAGE_VALID_RANGE[0] <= meta["V_min"] <= VOLTAGE_VALID_RANGE[1]):
        status = "WARN" if status != "FAIL" else status
        issues.append("voltage_min_outside_expected_range")

    if not (VOLTAGE_VALID_RANGE[0] <= meta["V_max"] <= VOLTAGE_VALID_RANGE[1]):
        status = "WARN" if status != "FAIL" else status
        issues.append("voltage_max_outside_expected_range")

    if not (TEMP_VALID_RANGE[0] <= meta["T_min"] <= TEMP_VALID_RANGE[1]):
        status = "WARN" if status != "FAIL" else status
        issues.append("temperature_min_outside_expected_range")

    if not (TEMP_VALID_RANGE[0] <= meta["T_max"] <= TEMP_VALID_RANGE[1]):
        status = "WARN" if status != "FAIL" else status
        issues.append("temperature_max_outside_expected_range")

    return status, ";".join(issues)


def create_cycle_summary_row(meta_row, prep_meta):
    row = dict(meta_row)
    row.update(prep_meta)
    status, issues = classify_cycle_status(prep_meta)
    row["prep_status"] = status
    row["prep_issues"] = issues
    return row


# ============================================================
# 5. Plotting
# ============================================================

def plot_cycle_diagnostic(df_cycle, summary_row, out_dir):
    batt = summary_row["battery_id"]
    fname = str(summary_row["filename"]).replace(".", "_")
    time_h = df_cycle["time_s"].to_numpy() / 3600.0

    fig, ax = plt.subplots(5, 1, figsize=(10, 11), sharex=True)

    ax[0].plot(time_h, df_cycle["Voltage_measured"], lw=1.2)
    ax[0].set_ylabel("Voltage (V)")
    ax[0].grid(alpha=0.3)

    ax[1].plot(time_h, df_cycle["Current_raw"], lw=1.0, label="Raw")
    ax[1].plot(time_h, df_cycle["Current_discharge"], "--", lw=1.0, label="Discharge")
    ax[1].set_ylabel("Current (A)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    ax[2].plot(time_h, df_cycle["Temperature_measured"], lw=1.0)
    ax[2].set_ylabel("Temp (°C)")
    ax[2].grid(alpha=0.3)

    ax[3].plot(time_h, df_cycle["SOC"]*100, lw=1.5)
    ax[3].set_ylabel("SOC (%)")
    ax[3].grid(alpha=0.3)

    ax[4].plot(df_cycle["SOC"]*100, df_cycle["Voltage_measured"], lw=1.2)
    ax[4].set_xlabel("SOC (%)")
    ax[4].set_ylabel("V vs SOC")
    ax[4].grid(alpha=0.3)

    title = (
        f"{batt} | {summary_row['filename']} | "
        f"SOH={summary_row['SOH']:.3f} | {summary_row['prep_status']}"
    )
    fig.suptitle(title, fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"diagnostic_{batt}_{fname}.png"),
                dpi=250, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, f"diagnostic_{batt}_{fname}.pdf"),
                dpi=250, bbox_inches="tight")
    plt.close()


def plot_battery_overview(df_all, summary_df, out_dir):
    # Voltage-SOC curves
    fig, ax = plt.subplots(figsize=(8, 5))

    for batt, g in df_all.groupby("battery_id"):
        # plot up to 10 cycles per battery for readability
        for _, cyc in list(g.groupby("cycle_index"))[:10]:
            ax.plot(cyc["SOC"]*100, cyc["Voltage_measured"], alpha=0.35, lw=0.9)

    ax.set_xlabel("SOC (%)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("NASA Voltage-SOC Curves Across Batteries")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "overview_voltage_soc_curves.png"),
                dpi=250, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "overview_voltage_soc_curves.pdf"),
                dpi=250, bbox_inches="tight")
    plt.close()

    # Capacity fade
    fig, ax = plt.subplots(figsize=(8, 5))
    for batt, g in summary_df.groupby("battery_id"):
        ax.plot(g["cycle_index"], g["Capacity"], marker="o", ms=3, lw=1.0, label=batt)

    ax.set_xlabel("Cycle index")
    ax.set_ylabel("Capacity (Ah)")
    ax.set_title("NASA Capacity Fade")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "overview_capacity_fade.png"),
                dpi=250, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "overview_capacity_fade.pdf"),
                dpi=250, bbox_inches="tight")
    plt.close()

    # SOH trend
    fig, ax = plt.subplots(figsize=(8, 5))
    for batt, g in summary_df.groupby("battery_id"):
        ax.plot(g["cycle_index"], g["SOH"], marker="o", ms=3, lw=1.0, label=batt)

    ax.set_xlabel("Cycle index")
    ax.set_ylabel("SOH")
    ax.set_title("NASA SOH Trend")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "overview_soh_trend.png"),
                dpi=250, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "overview_soh_trend.pdf"),
                dpi=250, bbox_inches="tight")
    plt.close()


# ============================================================
# 6. Cross-battery fold definition
# ============================================================

def create_leave_one_battery_out_folds(battery_ids):
    rows = []
    for test_batt in battery_ids:
        train_batts = [b for b in battery_ids if b != test_batt]
        rows.append({
            "fold_name": f"test_{test_batt}",
            "test_battery": test_batt,
            "train_batteries": "+".join(train_batts),
        })
    return pd.DataFrame(rows)


# ============================================================
# 7. Main preparation pipeline
# ============================================================

def prepare_nasa_clean_dataset():
    print("\n" + "█"*92)
    print(" Professional NASA Data Preparation Before Training")
    print("█"*92)

    print(f"Metadata: {METADATA_PATH}")
    print(f"Data dir : {DATA_DIR}")
    print(f"Batteries: {BATTERY_IDS}")
    print(f"MAX_CYCLES_PER_BATTERY: {MAX_CYCLES_PER_BATTERY}")
    print(f"RESAMPLE_N: {RESAMPLE_N}")

    metadata_df = load_metadata(
        METADATA_PATH,
        battery_ids=BATTERY_IDS,
        max_cycles_per_battery=MAX_CYCLES_PER_BATTERY
    )

    if metadata_df.empty:
        raise RuntimeError("No discharge cycles found after filtering metadata.")

    print(f"\nDischarge cycles selected: {len(metadata_df)}")
    print(metadata_df.groupby("battery_id").size())

    all_sample_dfs = []
    cycle_summary_rows = []
    first_cycle_plotted = set()

    for row_i, meta_row in metadata_df.iterrows():
        batt = str(meta_row["battery_id"])
        fname = str(meta_row["filename"])
        cap = safe_float(meta_row["Capacity"])
        q_nom = safe_float(meta_row["Q_nom"])
        soh = safe_float(meta_row["SOH"])

        # cycle_index within each battery based on sorted order.
        cycle_index = int(
            metadata_df.loc[:row_i]
            .query("battery_id == @batt")
            .shape[0] - 1
        )

        try:
            fpath = find_cycle_file(DATA_DIR, fname)
            raw_df = read_cycle_file(fpath)

            df_cycle, prep_meta = prepare_cycle_dataframe(
                raw_df=raw_df,
                battery_id=batt,
                filename=fname,
                cycle_index=cycle_index,
                capacity_ah=cap,
                q_nom_ah=q_nom,
                soh=soh,
                resample_n=RESAMPLE_N
            )

            summary_row = create_cycle_summary_row(meta_row.to_dict(), prep_meta)
            summary_row["cycle_index"] = cycle_index
            cycle_summary_rows.append(summary_row)

            all_sample_dfs.append(df_cycle)

            if batt not in first_cycle_plotted:
                plot_cycle_diagnostic(df_cycle, summary_row, PLOTS_DIR)
                first_cycle_plotted.add(batt)

            print(
                f"{batt} | {fname:<12} | {summary_row['prep_status']:<4} | "
                f"SOH={summary_row['SOH']:.3f} | "
                f"SOC {summary_row['SOC_start']:.3f}->{summary_row['SOC_end']:.3f} | "
                f"CapErr={summary_row['capacity_rel_error']*100:.2f}%"
            )

        except Exception as e:
            fail_row = meta_row.to_dict()
            fail_row["cycle_index"] = cycle_index
            fail_row["prep_status"] = "FAIL"
            fail_row["prep_issues"] = str(e)
            cycle_summary_rows.append(fail_row)
            print(f"{batt} | {fname:<12} | FAIL | {e}")

    if not all_sample_dfs:
        raise RuntimeError("No cycles were successfully prepared.")

    clean_df = pd.concat(all_sample_dfs, ignore_index=True)
    cycle_summary_df = pd.DataFrame(cycle_summary_rows)

    # Battery-level summary.
    battery_summary_rows = []
    for batt, g in cycle_summary_df.groupby("battery_id"):
        battery_summary_rows.append({
            "battery_id": batt,
            "n_cycles": int(len(g)),
            "n_pass": int((g["prep_status"] == "PASS").sum()),
            "n_warn": int((g["prep_status"] == "WARN").sum()),
            "n_fail": int((g["prep_status"] == "FAIL").sum()),
            "SOH_min": pd.to_numeric(g["SOH"], errors="coerce").min(),
            "SOH_max": pd.to_numeric(g["SOH"], errors="coerce").max(),
            "Capacity_min": pd.to_numeric(g["Capacity"], errors="coerce").min(),
            "Capacity_max": pd.to_numeric(g["Capacity"], errors="coerce").max(),
            "V_start_mean": pd.to_numeric(g["V_start"], errors="coerce").mean(),
            "V_end_mean": pd.to_numeric(g["V_end"], errors="coerce").mean(),
            "capacity_rel_error_mean": pd.to_numeric(g["capacity_rel_error"], errors="coerce").mean(),
        })

    battery_summary_df = pd.DataFrame(battery_summary_rows)
    folds_df = create_leave_one_battery_out_folds(sorted(clean_df["battery_id"].unique()))

    # Save outputs.
    cycle_summary_path = os.path.join(OUT_DIR, "nasa_prepared_cycle_summary.csv")
    battery_summary_path = os.path.join(OUT_DIR, "nasa_prepared_battery_summary.csv")
    folds_path = os.path.join(OUT_DIR, "nasa_leave_one_battery_out_folds.csv")

    cycle_summary_df.to_csv(cycle_summary_path, index=False)
    battery_summary_df.to_csv(battery_summary_path, index=False)
    folds_df.to_csv(folds_path, index=False)

    if SAVE_PER_BATTERY_FILES:
        for batt, g in clean_df.groupby("battery_id"):
            g.to_csv(os.path.join(OUT_DIR, f"nasa_clean_samples_{batt}.csv"), index=False)

    if SAVE_FULL_DATASET_CSV:
        clean_csv_path = os.path.join(OUT_DIR, "nasa_clean_samples_all.csv")
        clean_df.to_csv(clean_csv_path, index=False)
    else:
        clean_csv_path = None

    parquet_path = None
    if SAVE_PARQUET_IF_AVAILABLE:
        try:
            parquet_path = os.path.join(OUT_DIR, "nasa_clean_samples_all.parquet")
            clean_df.to_parquet(parquet_path, index=False)
        except Exception:
            parquet_path = None

    plot_battery_overview(clean_df, cycle_summary_df, PLOTS_DIR)

    # Save preparation configuration for reproducibility.
    config = {
        "METADATA_PATH": METADATA_PATH,
        "DATA_DIR": DATA_DIR,
        "BATTERY_IDS": BATTERY_IDS,
        "MAX_CYCLES_PER_BATTERY": MAX_CYCLES_PER_BATTERY,
        "RESAMPLE_N": RESAMPLE_N,
        "MIN_SAMPLES": MIN_SAMPLES,
        "MAX_DT_ALLOWED": MAX_DT_ALLOWED,
        "CAPACITY_REL_ERROR_WARN": CAPACITY_REL_ERROR_WARN,
        "CAPACITY_REL_ERROR_FAIL": CAPACITY_REL_ERROR_FAIL,
        "feature_columns": [
            "time_s", "time_norm", "dt_s",
            "Voltage_measured", "Current_discharge", "Temperature_measured",
            "SOC", "SOH", "SOH_delta",
            "Capacity_Ah", "Q_nom_Ah",
            "discharged_Ah", "cumulative_Wh",
            "dV_dt", "dI_dt", "R_dyn_approx",
            "cycle_index", "sample_index"
        ],
        "target_column": "SOC",
        "cross_battery_protocol": "leave_one_battery_out",
    }

    with open(os.path.join(OUT_DIR, "preparation_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\n" + "═"*92)
    print("PREPARATION COMPLETE")
    print("═"*92)
    print(f"Clean sample rows : {len(clean_df)}")
    print(f"Prepared cycles   : {len(cycle_summary_df)}")
    print(f"Cycle summary     : {cycle_summary_path}")
    print(f"Battery summary   : {battery_summary_path}")
    print(f"Folds             : {folds_path}")
    if clean_csv_path:
        print(f"Clean CSV         : {clean_csv_path}")
    if parquet_path:
        print(f"Clean Parquet     : {parquet_path}")
    print(f"Diagnostic plots  : {PLOTS_DIR}")

    print("\nBattery summary:")
    print(battery_summary_df.to_string(index=False))

    print("\nRecommended next step:")
    print("  Use nasa_clean_samples_all.csv or per-battery files as the only source for training.")
    print("  Do not recalculate SOC inside the training script.")
    print("  Use the fold file for leave-one-battery-out experiments.")

    return clean_df, cycle_summary_df, battery_summary_df, folds_df


if __name__ == "__main__":
    clean_df, cycle_summary_df, battery_summary_df, folds_df = prepare_nasa_clean_dataset()


# nasa_step1b_ocv_ecm_refinement_before_training.py
# ============================================================
# STEP 1B: Improved OCV / ECM Refinement Before Training
#
# Purpose:
#   Improve the NASA-calibrated battery model before EKF / DU-EKF:
#
#   1) Use prepared clean NASA data only.
#   2) Build monotonic pseudo-OCV(SOC) lookup table.
#   3) Constrain temperature coefficient kT to physically reasonable range.
#   4) Constrain R0, R1, C1 to more realistic ranges.
#   5) Save refined model JSON for the training script.
#
# Input:
#   nasa_prepared_clean_dataset/nasa_clean_samples_all.csv
#
# Output:
#   nasa_ocv_ecm_refined_results/
#       refined_battery_model.json
#       refined_ecm_parameters_by_battery.csv
#       refined_ocv_lookup_by_battery.csv
#       refined_voltage_fit_metrics_by_battery.csv
#       plots/
#
# Run:
#   python nasa_step1b_ocv_ecm_refinement_before_training.py
#
# Requirements:
#   pip install numpy pandas matplotlib scipy
# ============================================================

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    from scipy.optimize import differential_evolution, minimize
    from scipy.interpolate import PchipInterpolator
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# ============================================================
# 0. Configuration
# ============================================================

PREPARED_DIR = "nasa_prepared_clean_dataset"
CLEAN_CSV = os.path.join(PREPARED_DIR, "nasa_clean_samples_all.csv")

OUT_DIR = "nasa_ocv_ecm_refined_results"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

BATTERY_IDS = ["B0005", "B0006", "B0007", "B0018"]

# Use all cycles by default.
MAX_CYCLES_PER_BATTERY = None

# Optimization sample limit per battery.
MAX_OPT_POINTS_PER_BATTERY = 30000

# OCV lookup grid.
SOC_GRID = np.linspace(0.0, 1.0, 101)

# Physical constraints.
# NASA aged 18650 cells can show increased internal resistance, but avoid unrealistic compensation.
R0_BOUNDS = (0.005, 0.090)       # Ohm
R1_BOUNDS = (0.001, 0.150)       # Ohm
C1_BOUNDS = (200.0, 15000.0)     # Farad

# Li-ion OCV temperature coefficient is small. Use mV/°C range, not tens of mV/°C.
KT_BOUNDS = (-0.0008, 0.0008)    # V/°C

USE_TEMP_COEFF = True
T_REF = 25.0

RANDOM_SEED = 42


# ============================================================
# 1. Utilities
# ============================================================

def set_seed(seed=42):
    np.random.seed(seed)


def rmse(a, b):
    e = np.asarray(a) - np.asarray(b)
    return float(np.sqrt(np.mean(e**2)))


def mae(a, b):
    e = np.asarray(a) - np.asarray(b)
    return float(np.mean(np.abs(e)))


def r2_score_np(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def load_clean_dataset():
    if not os.path.exists(CLEAN_CSV):
        raise FileNotFoundError(
            f"Clean dataset not found: {CLEAN_CSV}\n"
            "Run nasa_professional_data_preparation_before_training.py first."
        )

    df = pd.read_csv(CLEAN_CSV)

    required = [
        "battery_id", "cycle_index", "sample_index",
        "Voltage_measured", "Current_discharge",
        "Temperature_measured", "SOC", "SOH", "dt_s"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in clean dataset: {missing}")

    df["battery_id"] = df["battery_id"].astype(str)
    df = df[df["battery_id"].isin(BATTERY_IDS)].copy()

    df["cycle_index"] = pd.to_numeric(df["cycle_index"], errors="coerce").astype(int)
    df["sample_index"] = pd.to_numeric(df["sample_index"], errors="coerce").astype(int)

    df = df.sort_values(["battery_id", "cycle_index", "sample_index"]).reset_index(drop=True)

    if MAX_CYCLES_PER_BATTERY is not None:
        parts = []
        for b, g in df.groupby("battery_id"):
            keep_cycles = sorted(g["cycle_index"].unique())[:MAX_CYCLES_PER_BATTERY]
            parts.append(g[g["cycle_index"].isin(keep_cycles)])
        df = pd.concat(parts, ignore_index=True)

    return df


def split_cycles(df_batt):
    cycles = []
    for cyc, g in df_batt.groupby("cycle_index", sort=True):
        g = g.sort_values("sample_index").reset_index(drop=True)
        cycles.append({
            "cycle_index": int(cyc),
            "SOC": g["SOC"].to_numpy(dtype=float),
            "V": g["Voltage_measured"].to_numpy(dtype=float),
            "I": g["Current_discharge"].to_numpy(dtype=float),
            "T": g["Temperature_measured"].to_numpy(dtype=float),
            "dt": g["dt_s"].to_numpy(dtype=float),
            "SOH": g["SOH"].to_numpy(dtype=float),
        })
    return cycles


# ============================================================
# 2. ECM model
# ============================================================

def simulate_vrc(I, dt, R1, C1):
    n = len(I)
    vrc = np.zeros(n, dtype=float)
    tau = max(R1*C1, 1e-9)

    for k in range(1, n):
        dtk = float(dt[k])
        if not np.isfinite(dtk) or dtk <= 0:
            dtk = 1.0
        dtk = min(dtk, 60.0)
        alpha = np.exp(-dtk/tau)
        vrc[k] = alpha*vrc[k-1] + (1-alpha)*R1*I[k-1]

    return vrc


def pchip_eval(soc_grid, ocv_grid, soc):
    soc = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)

    if SCIPY_AVAILABLE:
        f = PchipInterpolator(soc_grid, ocv_grid, extrapolate=True)
        return f(soc)

    # Fallback linear interpolation.
    return np.interp(soc, soc_grid, ocv_grid)


# ============================================================
# 3. Monotonic OCV estimation
# ============================================================

def monotonic_bin_average(soc, v_corr, soc_grid=SOC_GRID):
    soc = np.asarray(soc, dtype=float)
    v_corr = np.asarray(v_corr, dtype=float)

    mask = np.isfinite(soc) & np.isfinite(v_corr) & (soc >= 0) & (soc <= 1)
    soc = soc[mask]
    v_corr = v_corr[mask]

    # Robust binning.
    bin_edges = np.linspace(0, 1, len(soc_grid)+1)
    ocv = np.full(len(soc_grid), np.nan)

    for i, s in enumerate(soc_grid):
        if i == 0:
            lo, hi = 0.0, bin_edges[1]
        elif i == len(soc_grid)-1:
            lo, hi = bin_edges[-2], 1.0
        else:
            lo, hi = bin_edges[i], bin_edges[i+1]

        m = (soc >= lo) & (soc <= hi)
        if np.any(m):
            # median is robust against pulse and transient noise.
            ocv[i] = np.nanmedian(v_corr[m])

    # Fill missing by interpolation.
    valid = np.isfinite(ocv)
    if valid.sum() < 5:
        raise ValueError("Too few valid OCV bins.")

    ocv = np.interp(soc_grid, soc_grid[valid], ocv[valid])

    # Enforce monotonic increasing OCV with SOC.
    ocv_mono = np.maximum.accumulate(ocv)

    # Smooth lightly using moving average while preserving monotonicity.
    kernel = np.ones(5) / 5
    pad = np.pad(ocv_mono, (2, 2), mode="edge")
    ocv_smooth = np.convolve(pad, kernel, mode="valid")
    ocv_smooth = np.maximum.accumulate(ocv_smooth)

    # Preserve endpoints close to estimated values.
    ocv_smooth[0] = min(ocv_smooth[0], ocv_mono[0])
    ocv_smooth[-1] = max(ocv_smooth[-1], ocv_mono[-1])
    ocv_smooth = np.maximum.accumulate(ocv_smooth)

    return ocv_smooth


def estimate_ocv_lookup(cycles, R0, R1, C1, kT):
    soc_all = []
    vcorr_all = []

    for D in cycles:
        vrc = simulate_vrc(D["I"], D["dt"], R1, C1)
        v_corr = D["V"] + R0*D["I"] + vrc - kT*(D["T"] - T_REF)
        soc_all.append(D["SOC"])
        vcorr_all.append(v_corr)

    soc_all = np.concatenate(soc_all)
    vcorr_all = np.concatenate(vcorr_all)

    return monotonic_bin_average(soc_all, vcorr_all, SOC_GRID)


def estimate_voltage(cycles, R0, R1, C1, kT, ocv_grid):
    V_true_all, V_pred_all, SOC_all = [], [], []

    for D in cycles:
        vrc = simulate_vrc(D["I"], D["dt"], R1, C1)
        ocv = pchip_eval(SOC_GRID, ocv_grid, D["SOC"]) + kT*(D["T"] - T_REF)
        v_pred = ocv - R0*D["I"] - vrc

        V_true_all.append(D["V"])
        V_pred_all.append(v_pred)
        SOC_all.append(D["SOC"])

    return {
        "V_true": np.concatenate(V_true_all),
        "V_pred": np.concatenate(V_pred_all),
        "SOC": np.concatenate(SOC_all),
    }


def objective(params, cycles):
    if USE_TEMP_COEFF:
        R0, R1, C1, kT = params
    else:
        R0, R1, C1 = params
        kT = 0.0

    if R0 <= 0 or R1 <= 0 or C1 <= 0:
        return 1e6

    try:
        ocv_grid = estimate_ocv_lookup(cycles, R0, R1, C1, kT)
        res = estimate_voltage(cycles, R0, R1, C1, kT, ocv_grid)
        err = res["V_true"] - res["V_pred"]

        loss = np.sqrt(np.mean(err**2)) + 0.2*np.mean(np.abs(err))

        # Penalize unrealistic flat or too steep OCV.
        ocv_span = ocv_grid[-1] - ocv_grid[0]
        if ocv_span < 0.5 or ocv_span > 1.6:
            loss += 0.05

        # Penalize boundary solution mildly.
        if abs(R0 - R0_BOUNDS[1]) < 1e-4:
            loss += 0.01
        if abs(R1 - R1_BOUNDS[1]) < 1e-4:
            loss += 0.01

        return float(loss)

    except Exception:
        return 1e6


# ============================================================
# 4. Optimization
# ============================================================

def subsample_cycles(cycles, max_points):
    total = sum(len(D["SOC"]) for D in cycles)
    if total <= max_points:
        return cycles

    ratio = max_points / total
    step = max(1, int(round(1/ratio)))

    out = []
    for D in cycles:
        idx = np.arange(0, len(D["SOC"]), step)
        out.append({
            "cycle_index": D["cycle_index"],
            "SOC": D["SOC"][idx],
            "V": D["V"][idx],
            "I": D["I"][idx],
            "T": D["T"][idx],
            "dt": D["dt"][idx],
            "SOH": D["SOH"][idx],
        })
    return out


def identify_refined_parameters(battery_id, cycles):
    print(f"\nRefining OCV/ECM for {battery_id}")
    print(f"  cycles={len(cycles)} | samples={sum(len(D['SOC']) for D in cycles)}")

    opt_cycles = subsample_cycles(cycles, MAX_OPT_POINTS_PER_BATTERY)

    bounds = [
        R0_BOUNDS,
        R1_BOUNDS,
        C1_BOUNDS,
    ]
    if USE_TEMP_COEFF:
        bounds.append(KT_BOUNDS)

    if SCIPY_AVAILABLE:
        result_de = differential_evolution(
            func=lambda x: objective(x, opt_cycles),
            bounds=bounds,
            maxiter=35,
            popsize=10,
            tol=1e-5,
            seed=RANDOM_SEED,
            polish=False,
            workers=1,
            updating="immediate"
        )

        result_local = minimize(
            fun=lambda x: objective(x, opt_cycles),
            x0=result_de.x,
            method="Nelder-Mead",
            options={"maxiter": 250, "xatol": 1e-7, "fatol": 1e-7}
        )

        if result_local.fun < result_de.fun:
            best = result_local.x
            best_loss = float(result_local.fun)
        else:
            best = result_de.x
            best_loss = float(result_de.fun)
    else:
        rng = np.random.default_rng(RANDOM_SEED)
        best, best_loss = None, np.inf
        for _ in range(1500):
            cand = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
            val = objective(cand, opt_cycles)
            if val < best_loss:
                best, best_loss = cand, val

    if USE_TEMP_COEFF:
        R0, R1, C1, kT = best
    else:
        R0, R1, C1 = best
        kT = 0.0

    ocv_grid = estimate_ocv_lookup(cycles, R0, R1, C1, kT)
    fit = estimate_voltage(cycles, R0, R1, C1, kT, ocv_grid)

    metrics = {
        "battery_id": battery_id,
        "R0_ohm": float(R0),
        "R1_ohm": float(R1),
        "C1_F": float(C1),
        "tau_s": float(R1*C1),
        "kT_V_per_C": float(kT),
        "objective_loss": best_loss,
        "voltage_RMSE_V": rmse(fit["V_true"], fit["V_pred"]),
        "voltage_MAE_V": mae(fit["V_true"], fit["V_pred"]),
        "voltage_R2": r2_score_np(fit["V_true"], fit["V_pred"]),
        "ocv_min_V": float(np.min(ocv_grid)),
        "ocv_max_V": float(np.max(ocv_grid)),
        "ocv_span_V": float(ocv_grid[-1] - ocv_grid[0]),
        "n_cycles": len(cycles),
        "n_samples": int(sum(len(D["SOC"]) for D in cycles)),
    }

    print(
        f"  R0={R0:.5f} Ω | R1={R1:.5f} Ω | C1={C1:.1f} F | "
        f"tau={R1*C1:.1f} s | kT={kT:.6f} V/C"
    )
    print(
        f"  OCV span={metrics['ocv_span_V']:.3f} V | "
        f"Voltage RMSE={metrics['voltage_RMSE_V']:.5f} V | "
        f"R2={metrics['voltage_R2']:.4f}"
    )

    return metrics, ocv_grid, fit


# ============================================================
# 5. Global model
# ============================================================

def make_global_model(param_df, ocv_lookup_df):
    global_params = {
        "R0_ohm": float(param_df["R0_ohm"].median()),
        "R1_ohm": float(param_df["R1_ohm"].median()),
        "C1_F": float(param_df["C1_F"].median()),
        "tau_s": float(param_df["tau_s"].median()),
        "kT_V_per_C": float(param_df["kT_V_per_C"].median()),
    }

    ocv_cols = [c for c in ocv_lookup_df.columns if c.startswith("ocv_")]
    ocv_mat = ocv_lookup_df[ocv_cols].to_numpy(dtype=float)
    global_ocv = np.median(ocv_mat, axis=0)
    global_ocv = np.maximum.accumulate(global_ocv)

    return global_params, global_ocv


# ============================================================
# 6. Plotting
# ============================================================

def plot_fit(battery_id, fit, out_dir):
    V_true = fit["V_true"]
    V_pred = fit["V_pred"]
    SOC = fit["SOC"]

    n = len(V_true)
    step = max(1, n//8000)
    idx = np.arange(0, n, step)

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

    ax[0].plot(V_true[idx], lw=1.0, label="Measured")
    ax[0].plot(V_pred[idx], lw=1.0, label="Refined ECM")
    ax[0].set_ylabel("Voltage (V)")
    ax[0].set_title(f"{battery_id}: Refined Voltage Fit")
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    ax[1].plot((V_pred[idx] - V_true[idx])*1000, lw=1.0)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel("Error (mV)")
    ax[1].set_xlabel("Sample")
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{battery_id}_refined_voltage_fit.png"), dpi=250, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, f"{battery_id}_refined_voltage_fit.pdf"), dpi=250, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(SOC[idx]*100, V_true[idx], s=2, alpha=0.25, label="Measured")
    ax.scatter(SOC[idx]*100, V_pred[idx], s=2, alpha=0.25, label="Refined ECM")
    ax.set_xlabel("SOC (%)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"{battery_id}: V-SOC Refined Fit")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{battery_id}_refined_voltage_soc_fit.png"), dpi=250, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, f"{battery_id}_refined_voltage_soc_fit.pdf"), dpi=250, bbox_inches="tight")
    plt.close()


def plot_ocv_lookup(ocv_lookup_df, global_ocv, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))

    for _, row in ocv_lookup_df.iterrows():
        ocv = np.array([row[f"ocv_{i:03d}"] for i in range(len(SOC_GRID))], dtype=float)
        ax.plot(SOC_GRID*100, ocv, lw=1.3, label=row["battery_id"])

    ax.plot(SOC_GRID*100, global_ocv, "k--", lw=2.0, label="Global median")
    ax.set_xlabel("SOC (%)")
    ax.set_ylabel("OCV / pseudo-OCV (V)")
    ax.set_title("Refined Monotonic OCV Lookup")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "refined_ocv_lookup_curves.png"), dpi=250, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "refined_ocv_lookup_curves.pdf"), dpi=250, bbox_inches="tight")
    plt.close()


def plot_parameter_summary(param_df, out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    axes[0].bar(param_df["battery_id"], param_df["R0_ohm"])
    axes[0].set_title("R0")
    axes[0].set_ylabel("Ω")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(param_df["battery_id"], param_df["R1_ohm"])
    axes[1].set_title("R1")
    axes[1].set_ylabel("Ω")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(param_df["battery_id"], param_df["C1_F"])
    axes[2].set_title("C1")
    axes[2].set_ylabel("F")
    axes[2].grid(axis="y", alpha=0.3)

    axes[3].bar(param_df["battery_id"], param_df["kT_V_per_C"]*1000)
    axes[3].set_title("kT")
    axes[3].set_ylabel("mV/°C")
    axes[3].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "refined_parameter_summary.png"), dpi=250, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "refined_parameter_summary.pdf"), dpi=250, bbox_inches="tight")
    plt.close()


# ============================================================
# 7. Save model
# ============================================================

def save_refined_model(param_df, ocv_lookup_df, global_params, global_ocv):
    model = {
        "description": "Refined NASA-calibrated first-order Thevenin ECM with monotonic OCV lookup",
        "model_type": "Thevenin_1RC_monotonic_OCV_lookup",
        "soc_grid": [float(x) for x in SOC_GRID],
        "temperature_reference_C": T_REF,
        "use_temperature_coefficient": USE_TEMP_COEFF,
        "global": {
            **global_params,
            "ocv_lookup": [float(x) for x in global_ocv],
        },
        "by_battery": {}
    }

    for _, p in param_df.iterrows():
        batt = p["battery_id"]
        row = ocv_lookup_df[ocv_lookup_df["battery_id"] == batt].iloc[0]
        ocv = [float(row[f"ocv_{i:03d}"]) for i in range(len(SOC_GRID))]

        model["by_battery"][batt] = {
            "R0_ohm": float(p["R0_ohm"]),
            "R1_ohm": float(p["R1_ohm"]),
            "C1_F": float(p["C1_F"]),
            "tau_s": float(p["tau_s"]),
            "kT_V_per_C": float(p["kT_V_per_C"]),
            "ocv_lookup": ocv,
            "voltage_RMSE_V": float(p["voltage_RMSE_V"]),
            "voltage_MAE_V": float(p["voltage_MAE_V"]),
            "voltage_R2": float(p["voltage_R2"]),
        }

    path = os.path.join(OUT_DIR, "refined_battery_model.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)

    return path


# ============================================================
# 8. Main
# ============================================================

def main():
    print("\n" + "█"*90)
    print(" STEP 1B: Refined OCV / ECM Identification Before Training")
    print("█"*90)

    set_seed(RANDOM_SEED)

    print(f"Clean dataset: {CLEAN_CSV}")
    print(f"Scipy available: {SCIPY_AVAILABLE}")
    print(f"R0 bounds: {R0_BOUNDS}")
    print(f"R1 bounds: {R1_BOUNDS}")
    print(f"C1 bounds: {C1_BOUNDS}")
    print(f"kT bounds: {KT_BOUNDS}")
    print(f"OCV grid points: {len(SOC_GRID)}")

    df = load_clean_dataset()
    print(f"\nLoaded samples: {len(df)}")
    print(df["battery_id"].value_counts().sort_index())

    param_rows = []
    ocv_rows = []
    metric_rows = []

    for batt in BATTERY_IDS:
        df_b = df[df["battery_id"] == batt].copy()
        if df_b.empty:
            print(f"Skipping {batt}: no samples.")
            continue

        cycles = split_cycles(df_b)
        metrics, ocv_grid, fit = identify_refined_parameters(batt, cycles)

        param_rows.append(metrics)
        metric_rows.append({
            "battery_id": batt,
            "voltage_RMSE_V": metrics["voltage_RMSE_V"],
            "voltage_MAE_V": metrics["voltage_MAE_V"],
            "voltage_R2": metrics["voltage_R2"],
            "ocv_span_V": metrics["ocv_span_V"],
            "n_cycles": metrics["n_cycles"],
            "n_samples": metrics["n_samples"],
        })

        row = {"battery_id": batt}
        for i, val in enumerate(ocv_grid):
            row[f"ocv_{i:03d}"] = float(val)
        ocv_rows.append(row)

        plot_fit(batt, fit, PLOTS_DIR)

    if not param_rows:
        raise RuntimeError("No refined model was identified.")

    param_df = pd.DataFrame(param_rows)
    ocv_lookup_df = pd.DataFrame(ocv_rows)
    metrics_df = pd.DataFrame(metric_rows)

    global_params, global_ocv = make_global_model(param_df, ocv_lookup_df)

    global_row = dict(global_params)
    for i, val in enumerate(global_ocv):
        global_row[f"ocv_{i:03d}"] = float(val)
    global_df = pd.DataFrame([global_row])

    param_path = os.path.join(OUT_DIR, "refined_ecm_parameters_by_battery.csv")
    global_path = os.path.join(OUT_DIR, "refined_ecm_parameters_global.csv")
    ocv_path = os.path.join(OUT_DIR, "refined_ocv_lookup_by_battery.csv")
    metrics_path = os.path.join(OUT_DIR, "refined_voltage_fit_metrics_by_battery.csv")

    param_df.to_csv(param_path, index=False)
    global_df.to_csv(global_path, index=False)
    ocv_lookup_df.to_csv(ocv_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    json_path = save_refined_model(param_df, ocv_lookup_df, global_params, global_ocv)

    plot_ocv_lookup(ocv_lookup_df, global_ocv, PLOTS_DIR)
    plot_parameter_summary(param_df, PLOTS_DIR)

    print("\n" + "═"*90)
    print("REFINEMENT COMPLETE")
    print("═"*90)

    print("\nRefined parameters:")
    print(param_df[[
        "battery_id", "R0_ohm", "R1_ohm", "C1_F", "tau_s",
        "kT_V_per_C", "voltage_RMSE_V", "voltage_MAE_V", "voltage_R2", "ocv_span_V"
    ]].to_string(index=False))

    print("\nGlobal refined model:")
    for k, v in global_params.items():
        print(f"  {k}: {v}")

    print("\nSaved:")
    print(f"  {param_path}")
    print(f"  {global_path}")
    print(f"  {ocv_path}")
    print(f"  {metrics_path}")
    print(f"  {json_path}")
    print(f"  plots: {PLOTS_DIR}")

    print("\nNext step:")
    print("  Update training script to load:")
    print(f"    {json_path}")
    print("  and use the refined OCV lookup + R0/R1/C1 instead of synthetic BatteryParams.")

    return param_df, ocv_lookup_df, metrics_df


if __name__ == "__main__":
    param_df, ocv_lookup_df, metrics_df = main()


# nasa_training_with_current_identified_model.py
# ============================================================
# NASA Training using CURRENT identified/refined ECM results
#
# Uses:
#   1) Clean prepared dataset:
#        nasa_prepared_clean_dataset/nasa_clean_samples_all.csv
#
#   2) Current identified/refined battery model:
#        Preferred:
#          nasa_ocv_ecm_refined_results/refined_battery_model.json
#        Fallback:
#          nasa_parameter_identification_results/calibrated_battery_model.json
#
# Modes:
#   FAST_MODE=True
#       B0005 only, 70/30 cycle-level split.
#
#   FAST_MODE=False
#       Leave-One-Battery-Out cross-battery validation.
#
# Models:
#   EKF using identified ECM
#   DU-EKF using identified ECM + learned Q,R
#   DU-EKF + Residual BiLSTM
#
# IMPORTANT:
#   This code does not recalculate SOC/SOH.
#   It uses SOC and SOH from the prepared clean dataset.
#
# Run:
#   python nasa_training_with_current_identified_model.py
# ============================================================

import os
import json
import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# 0. Configuration
# ============================================================

CLEAN_CSV = r"nasa_prepared_clean_dataset\nasa_clean_samples_all.csv"

REFINED_MODEL_JSON = r"nasa_ocv_ecm_refined_results\refined_battery_model.json"
CALIBRATED_MODEL_JSON = r"nasa_parameter_identification_results\calibrated_battery_model.json"

OUT_DIR = "nasa_training_current_identified_model_results"
os.makedirs(OUT_DIR, exist_ok=True)

FAST_MODE = False

BATTERY_IDS = ["B0005", "B0006", "B0007", "B0018"]
FAST_BATTERY_ID = "B0005"
FAST_TRAIN_RATIO = 0.70

SEED = 42

SOC0_EST = 1.0
VRC0_EST = 0.0

Q_EKF = np.diag([1e-6, 1e-7]).astype(np.float64)
R_EKF = 1e-4

if FAST_MODE:
    DU_EPOCHS = 60
    RES_EPOCHS = 30
    QR_TRAIN_CYCLES = 25
    WINDOW = 80
    HIDDEN = 48
else:
    DU_EPOCHS = 140
    RES_EPOCHS = 90
    QR_TRAIN_CYCLES = 80
    WINDOW = 100
    HIDDEN = 64

BATCH_SIZE = 256
VAL_BATCH_SIZE = 128

RESIDUAL_CLIP = 0.04
PRED_CLIP = 0.035

BASE_FEATURES = [
    "Voltage_measured",
    "Current_discharge",
    "Temperature_measured",
    "SOH",
    "SOH_delta",
    "time_norm",
    "dV_dt",
    "dI_dt",
    "discharged_Ah",
    "cumulative_Wh",
    "R_dyn_approx",
]

RESIDUAL_FEATURES = BASE_FEATURES + ["SOC_DU", "V_EKF_pred", "V_EKF_error"]


# ============================================================
# 1. Utilities
# ============================================================

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rmse_np(a, b):
    e = np.asarray(a) - np.asarray(b)
    return float(np.sqrt(np.mean(e**2)))


def mae_np(a, b):
    e = np.asarray(a) - np.asarray(b)
    return float(np.mean(np.abs(e)))


def compute_metrics(est, true):
    e = np.asarray(est) - np.asarray(true)
    return {
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "MAE": float(np.mean(np.abs(e))),
        "MaxAE": float(np.max(np.abs(e))),
        "Bias": float(np.mean(e)),
        "P95": float(np.percentile(np.abs(e), 95)),
    }


# ============================================================
# 2. Load current identified battery model
# ============================================================

class IdentifiedBatteryModel:
    def __init__(self, model_json):
        with open(model_json, "r", encoding="utf-8") as f:
            self.model = json.load(f)

        self.model_json = model_json
        self.global_model = self.model["global"]
        self.by_battery = self.model.get("by_battery", {})

        self.is_lookup = "soc_grid" in self.model and "ocv_lookup" in self.global_model
        self.is_poly = "ocv_coeff" in self.global_model

        if self.is_lookup:
            self.soc_grid = np.array(self.model["soc_grid"], dtype=float)
        else:
            self.soc_grid = None

        self.t_ref = float(self.model.get("temperature_reference_C", 25.0))

    def get_params(self, battery_id=None, use_battery_specific=True):
        if use_battery_specific and battery_id in self.by_battery:
            return self.by_battery[battery_id]
        return self.global_model

    def ocv(self, soc, battery_id=None, use_battery_specific=True):
        p = self.get_params(battery_id, use_battery_specific)
        soc = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)

        if "ocv_lookup" in p:
            ocv_lookup = np.array(p["ocv_lookup"], dtype=float)
            return np.interp(soc, self.soc_grid, ocv_lookup)

        if "ocv_coeff" in p:
            coeff = np.array(p["ocv_coeff"], dtype=float)
            return np.polyval(coeff, soc)

        raise KeyError("Model JSON has neither ocv_lookup nor ocv_coeff.")

    def docv_dsoc(self, soc, battery_id=None, use_battery_specific=True):
        d = 1e-5
        sp = np.clip(float(soc) + d, 0.001, 0.999)
        sm = np.clip(float(soc) - d, 0.001, 0.999)
        return float((self.ocv(sp, battery_id, use_battery_specific)
                      - self.ocv(sm, battery_id, use_battery_specific)) / (sp - sm))

    def get_ecm(self, battery_id=None, use_battery_specific=True):
        p = self.get_params(battery_id, use_battery_specific)
        return {
            "R0": float(p["R0_ohm"]),
            "R1": float(p["R1_ohm"]),
            "C1": float(p["C1_F"]),
            "kT": float(p.get("kT_V_per_C", 0.0)),
        }


def choose_model_json():
    if os.path.exists(REFINED_MODEL_JSON):
        return REFINED_MODEL_JSON
    if os.path.exists(CALIBRATED_MODEL_JSON):
        return CALIBRATED_MODEL_JSON
    raise FileNotFoundError(
        "No identified battery model was found.\n"
        f"Missing: {REFINED_MODEL_JSON}\n"
        f"Missing: {CALIBRATED_MODEL_JSON}"
    )


# ============================================================
# 3. Clean dataset
# ============================================================

def load_clean_dataset():
    if not os.path.exists(CLEAN_CSV):
        raise FileNotFoundError(f"Missing clean dataset: {CLEAN_CSV}")

    df = pd.read_csv(CLEAN_CSV)
    required = [
        "battery_id", "cycle_index", "sample_index",
        "Voltage_measured", "Current_discharge", "Temperature_measured",
        "SOC", "SOH", "SOH_delta", "time_norm", "dt_s"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Clean dataset missing columns: {missing}")

    df["battery_id"] = df["battery_id"].astype(str)
    df = df[df["battery_id"].isin(BATTERY_IDS)].copy()
    df["cycle_index"] = pd.to_numeric(df["cycle_index"], errors="coerce").astype(int)
    df["sample_index"] = pd.to_numeric(df["sample_index"], errors="coerce").astype(int)
    df = df.sort_values(["battery_id", "cycle_index", "sample_index"]).reset_index(drop=True)

    return df


def dataframe_to_cycles(df):
    cycles = []

    for (batt, cyc), g in df.groupby(["battery_id", "cycle_index"], sort=True):
        g = g.sort_values("sample_index").reset_index(drop=True)

        D = {
            "battery_id": str(batt),
            "cycle_index": int(cyc),
            "filename": str(g["filename"].iloc[0]) if "filename" in g.columns else f"{batt}_{cyc}",
            "V_meas": g["Voltage_measured"].to_numpy(dtype=np.float32),
            "I_meas": g["Current_discharge"].to_numpy(dtype=np.float32),
            "T": g["Temperature_measured"].to_numpy(dtype=np.float32),
            "SOC_true": g["SOC"].to_numpy(dtype=np.float32),
            "SOH_vec": g["SOH"].to_numpy(dtype=np.float32),
            "dt_s": g["dt_s"].to_numpy(dtype=np.float32),
        }

        for f in BASE_FEATURES:
            if f in g.columns:
                D[f] = g[f].to_numpy(dtype=np.float32)
            else:
                D[f] = np.zeros(len(g), dtype=np.float32)

        cycles.append(D)

    return cycles


def split_fast_70_30(cycles):
    batt_cycles = [c for c in cycles if c["battery_id"] == FAST_BATTERY_ID]
    batt_cycles = sorted(batt_cycles, key=lambda x: x["cycle_index"])

    n_train = int(round(len(batt_cycles) * FAST_TRAIN_RATIO))
    n_train = max(1, min(n_train, len(batt_cycles)-1))

    return batt_cycles[:n_train], batt_cycles[n_train:]


def make_lobo_folds(cycles):
    folds = []
    for test_batt in BATTERY_IDS:
        train = [c for c in cycles if c["battery_id"] != test_batt]
        test = [c for c in cycles if c["battery_id"] == test_batt]
        if train and test:
            folds.append((test_batt, train, test))
    return folds


def split_train_val(train_cycles, val_fraction=0.20):
    train_cycles = sorted(train_cycles, key=lambda x: (x["battery_id"], x["cycle_index"]))
    n_val = max(1, int(round(len(train_cycles) * val_fraction)))
    return train_cycles[:-n_val], train_cycles[-n_val:]


# ============================================================
# 4. EKF with identified ECM
# ============================================================

def run_identified_ekf(D, model, Q, R, use_battery_specific=True):
    I = D["I_meas"]
    V = D["V_meas"]
    T = D["T"]
    dt_arr = D["dt_s"]
    batt = D["battery_id"]

    prm = model.get_ecm(batt, use_battery_specific)
    R0 = prm["R0"]
    R1 = prm["R1"]
    C1 = prm["C1"]
    kT = prm["kT"]

    n = len(I)
    x = np.array([SOC0_EST, VRC0_EST], dtype=np.float64)
    P = np.diag([0.03, 0.003]).astype(np.float64)

    soc_out = np.zeros(n, dtype=np.float64)
    v_pred_out = np.zeros(n, dtype=np.float64)
    innov_out = np.zeros(n, dtype=np.float64)

    tau = max(R1*C1, 1e-9)

    for k in range(n):
        dt = float(dt_arr[k])
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0
        dt = min(dt, 60.0)

        alpha = np.exp(-dt / tau)

        F = np.array([[1.0, 0.0], [0.0, alpha]], dtype=np.float64)

        # Use prepared SOC convention; Q_eff normalized by SOH for physics propagation.
        soh = float(D["SOH_vec"][k])
        Q_eff = max(0.5, 2.0 * soh)

        xp = np.array([
            np.clip(x[0] - dt/(Q_eff*3600.0)*float(I[k]), 0.01, 0.99),
            alpha*x[1] + (1.0-alpha)*R1*float(I[k])
        ], dtype=np.float64)

        Pp = F @ P @ F.T + Q

        ocv = float(model.ocv(xp[0], batt, use_battery_specific))
        Vp = ocv + kT*(float(T[k]) - model.t_ref) - R0*float(I[k]) - xp[1]

        H = np.array([[model.docv_dsoc(xp[0], batt, use_battery_specific), -1.0]], dtype=np.float64)

        S = float(H @ Pp @ H.T) + R + 1e-12
        K = (Pp @ H.T) / S

        innovation = float(V[k]) - Vp
        x = xp + K.flatten()*innovation
        x[0] = np.clip(x[0], 0.01, 0.99)

        A = np.eye(2) - K @ H
        P = A @ Pp @ A.T + K*R@K.T
        P = 0.5*(P + P.T) + 1e-12*np.eye(2)

        soc_out[k] = x[0]
        v_pred_out[k] = Vp
        innov_out[k] = innovation

    return soc_out.astype(np.float32), v_pred_out.astype(np.float32), innov_out.astype(np.float32)


def train_duekf_qr(train_cycles, val_cycles, model, use_battery_specific=True,
                   n_epochs=DU_EPOCHS, lr=0.035, patience=15,
                   max_train_cycles=QR_TRAIN_CYCLES):

    theta = np.array([-11.5, -16.1, -7.6], dtype=np.float64)
    m_adam = np.zeros(3)
    v_adam = np.zeros(3)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    delta = 1e-4

    train_subset = train_cycles[:min(max_train_cycles, len(train_cycles))]
    val_subset = val_cycles if len(val_cycles) else train_subset[-max(1, len(train_subset)//4):]

    best_val = np.inf
    best_theta = theta.copy()
    bad = 0
    hist = []

    def loss_for_cases(cases, th):
        Q = np.diag([np.exp(th[0]), np.exp(th[1])]).astype(np.float64)
        R = float(np.exp(th[2]))
        losses = []
        for D in cases:
            est, _, _ = run_identified_ekf(D, model, Q, R, use_battery_specific)
            losses.append(np.mean((est - D["SOC_true"])**2))
        return float(np.mean(losses))

    for ep in range(1, n_epochs+1):
        tr_loss = loss_for_cases(train_subset, theta)

        grad = np.zeros(3)
        for p in range(3):
            thp = theta.copy()
            thp[p] += delta
            grad[p] = (loss_for_cases(train_subset, thp) - tr_loss) / delta

        m_adam = beta1*m_adam + (1-beta1)*grad
        v_adam = beta2*v_adam + (1-beta2)*(grad**2)
        mh = m_adam/(1-beta1**ep)
        vh = v_adam/(1-beta2**ep)
        theta = theta - lr*mh/(np.sqrt(vh)+eps)

        val_loss = loss_for_cases(val_subset, theta)

        hist.append({
            "epoch": ep,
            "train_rmse": math.sqrt(tr_loss),
            "val_rmse": math.sqrt(val_loss),
            "q_soc": float(np.exp(theta[0])),
            "q_vrc": float(np.exp(theta[1])),
            "r": float(np.exp(theta[2])),
        })

        if ep == 1 or ep % 5 == 0:
            print(f"    DU-EKF Epoch {ep:03d}/{n_epochs} | "
                  f"Train={math.sqrt(tr_loss)*100:.4f}% | Val={math.sqrt(val_loss)*100:.4f}%")

        if val_loss < best_val:
            best_val = val_loss
            best_theta = theta.copy()
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            print(f"    DU-EKF early stopping at epoch {ep}")
            break

    Q = np.diag([np.exp(best_theta[0]), np.exp(best_theta[1])]).astype(np.float64)
    R = float(np.exp(best_theta[2]))

    return Q, R, pd.DataFrame(hist)


def add_du_features(cycles, model, Q_du, R_du, use_battery_specific=True):
    out = []
    for D in cycles:
        C = dict(D)
        soc_du, v_pred, innov = run_identified_ekf(C, model, Q_du, R_du, use_battery_specific)
        C["SOC_DU"] = soc_du
        C["V_EKF_pred"] = v_pred
        C["V_EKF_error"] = innov
        out.append(C)
    return out


# ============================================================
# 5. Residual BiLSTM
# ============================================================

def build_windows(Fn, y, window):
    pad = np.tile(Fn[0:1], (window-1, 1))
    Fp = np.vstack([pad, Fn])
    return np.stack([Fp[i:i+window] for i in range(len(y))], axis=0).astype(np.float32)


def stack_residual_dataset(cycles, feature_names, window, mu=None, sig=None):
    all_F = np.concatenate([np.vstack([D[f] for f in feature_names]).T for D in cycles], axis=0)

    if mu is None:
        mu = all_F.mean(axis=0)
        sig = all_F.std(axis=0)
        sig[sig < 1e-12] = 1.0

    X_list, Y_list = [], []
    for D in cycles:
        F = np.vstack([D[f] for f in feature_names]).T
        Fn = (F - mu) / sig
        residual = np.clip(D["SOC_true"] - D["SOC_DU"], -RESIDUAL_CLIP, RESIDUAL_CLIP).astype(np.float32)
        X_list.append(build_windows(Fn, residual, window))
        Y_list.append(residual)

    return np.concatenate(X_list).astype(np.float32), np.concatenate(Y_list).astype(np.float32), mu, sig


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, h):
        w = torch.softmax(self.attn(h), dim=1)
        return (h*w).sum(dim=1)


class ResidualBiLSTMNet(nn.Module):
    def __init__(self, n_feat, hidden=48):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_feat,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.30
        )
        self.attn = AttentionLayer(hidden*2)
        self.fc = nn.Sequential(
            nn.Linear(hidden*2, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1),
            nn.Tanh()
        )

    def forward(self, x):
        h, _ = self.lstm(x)
        c = self.attn(h)
        return self.fc(c).squeeze(-1) * PRED_CLIP


def predict_in_batches(net, X, device, batch_size=VAL_BATCH_SIZE):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=device)
            out.append(net(xb).detach().cpu())
    return torch.cat(out).numpy()


def train_residual_net(net, Xtr, Ytr, Xvl, Yvl, device, epochs=RES_EPOCHS, lr=5e-4, patience=8):
    net.to(device)
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    crit = nn.SmoothL1Loss(beta=0.002)

    loader = DataLoader(
        TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                      torch.tensor(Ytr, dtype=torch.float32)),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    best = np.inf
    best_state = None
    bad = 0
    hist = []

    for ep in range(1, epochs+1):
        net.train()
        total = 0.0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            pred = net(xb)
            loss = crit(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total += loss.item()

        pred_v = predict_in_batches(net, Xvl, device)
        val_rmse = float(np.sqrt(np.mean((pred_v - Yvl)**2)))
        train_loss = total / max(1, len(loader))

        hist.append({"epoch": ep, "train_loss": train_loss, "val_residual_rmse": val_rmse})

        if ep == 1 or ep % 5 == 0:
            print(f"    Residual Epoch {ep:03d}/{epochs} | "
                  f"Train loss={train_loss:.6e} | Val residual RMSE={val_rmse*100:.4f}%")

        if val_rmse < best:
            best = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            print(f"    Residual early stopping at epoch {ep}")
            break

    if best_state is not None:
        net.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return net, pd.DataFrame(hist)


def predict_residual_cycle(net, D, mu, sig, device):
    F = np.vstack([D[f] for f in RESIDUAL_FEATURES]).T
    Fn = (F - mu) / sig
    X = build_windows(Fn, D["SOC_true"], WINDOW)
    return np.clip(predict_in_batches(net, X, device), -PRED_CLIP, PRED_CLIP)


# ============================================================
# 6. Evaluation / plots
# ============================================================

def evaluate(cycles, model, Q_du, R_du, residual_model=None, mu=None, sig=None, device=None):
    rows = []
    pred_packs = []

    for D in cycles:
        ekf, _, _ = run_identified_ekf(D, model, Q_EKF, R_EKF)
        du, v_pred, innov = run_identified_ekf(D, model, Q_du, R_du)

        Dtmp = dict(D)
        Dtmp["SOC_DU"] = du
        Dtmp["V_EKF_pred"] = v_pred
        Dtmp["V_EKF_error"] = innov

        if residual_model is not None:
            res = predict_residual_cycle(residual_model, Dtmp, mu, sig, device)
            hyb = np.clip(du + res, 0.01, 0.99)
        else:
            res = np.zeros_like(du)
            hyb = du.copy()

        ests = {"EKF": ekf, "DU_EKF": du, "DU_EKF_RES": hyb}

        for method, est in ests.items():
            m = compute_metrics(est, D["SOC_true"])
            rows.append({
                "battery_id": D["battery_id"],
                "cycle_index": D["cycle_index"],
                "filename": D["filename"],
                "method": method,
                **m
            })

        pred_packs.append({
            "D": D,
            "EKF": ekf,
            "DU_EKF": du,
            "DU_EKF_RES": hyb,
            "residual": res,
        })

    return pd.DataFrame(rows), pred_packs


def summarize(metrics_df):
    out = []
    for method, g in metrics_df.groupby("method"):
        out.append({
            "method": method,
            "RMSE_mean": g["RMSE"].mean(),
            "RMSE_std": g["RMSE"].std(ddof=0),
            "MAE_mean": g["MAE"].mean(),
            "MaxAE_mean": g["MaxAE"].mean(),
            "Bias_mean": g["Bias"].mean(),
            "P95_mean": g["P95"].mean(),
        })
    return pd.DataFrame(out).sort_values("method").reset_index(drop=True)


def print_summary(summary_df, title):
    print("\n" + "═"*80)
    print(title)
    print("═"*80)
    print(summary_df.to_string(index=False, formatters={
        "RMSE_mean": lambda x: f"{x*100:.4f}%",
        "RMSE_std": lambda x: f"{x*100:.4f}%",
        "MAE_mean": lambda x: f"{x*100:.4f}%",
        "MaxAE_mean": lambda x: f"{x*100:.4f}%",
        "Bias_mean": lambda x: f"{x*100:.4f}%",
        "P95_mean": lambda x: f"{x*100:.4f}%",
    }))


def plot_example(pred_pack, out_prefix):
    D = pred_pack["D"]
    x = np.arange(len(D["SOC_true"]))

    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ax[0].plot(x, D["SOC_true"]*100, "k", lw=2, label="True")
    ax[0].plot(x, pred_pack["EKF"]*100, "--", label="EKF")
    ax[0].plot(x, pred_pack["DU_EKF"]*100, "-.", label="DU-EKF")
    ax[0].plot(x, pred_pack["DU_EKF_RES"]*100, label="DU-EKF+Residual")
    ax[0].set_ylabel("SOC (%)")
    ax[0].set_title(f"{D['battery_id']} | Cycle {D['cycle_index']} | {D['filename']}")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)

    ax[1].plot(x, (pred_pack["EKF"]-D["SOC_true"])*100, "--", label="EKF")
    ax[1].plot(x, (pred_pack["DU_EKF"]-D["SOC_true"])*100, "-.", label="DU-EKF")
    ax[1].plot(x, (pred_pack["DU_EKF_RES"]-D["SOC_true"])*100, label="DU-EKF+Residual")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel("Error (%)")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)

    ax[2].plot(x, pred_pack["residual"]*100)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_ylabel("Residual (%)")
    ax[2].set_xlabel("Sample")
    ax[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_prefix + ".png", dpi=250, bbox_inches="tight")
    plt.savefig(out_prefix + ".pdf", dpi=250, bbox_inches="tight")
    plt.close()


def plot_summary(summary_df, title, out_prefix):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary_df["method"], summary_df["RMSE_mean"]*100)
    ax.set_ylabel("Mean RMSE (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_prefix + ".png", dpi=250, bbox_inches="tight")
    plt.savefig(out_prefix + ".pdf", dpi=250, bbox_inches="tight")
    plt.close()


# ============================================================
# 7. Modes
# ============================================================

def run_fast(cycles, model, device):
    print("\n" + "█"*80)
    print("FAST MODE | Current Identified Model | One Battery 70/30")
    print("█"*80)

    out_dir = os.path.join(OUT_DIR, "fast_70_30")
    os.makedirs(out_dir, exist_ok=True)

    train_cycles, val_cycles = split_fast_70_30(cycles)
    print(f"Battery={FAST_BATTERY_ID} | Train={len(train_cycles)} | Val={len(val_cycles)}")

    print("\nTraining DU-EKF Q,R...")
    Q_du, R_du, hist_du = train_duekf_qr(train_cycles, val_cycles, model)
    hist_du.to_csv(os.path.join(out_dir, "history_duekf.csv"), index=False)
    print(f"Q_du=diag([{Q_du[0,0]:.3e}, {Q_du[1,1]:.3e}]) | R_du={R_du:.3e}")

    train_aug = add_du_features(train_cycles, model, Q_du, R_du)
    val_aug = add_du_features(val_cycles, model, Q_du, R_du)

    print("\nPreparing residual data...")
    Xtr, Ytr, mu, sig = stack_residual_dataset(train_aug, RESIDUAL_FEATURES, WINDOW)
    Xvl, Yvl, _, _ = stack_residual_dataset(val_aug, RESIDUAL_FEATURES, WINDOW, mu, sig)
    print(f"Xtr={Xtr.shape}, Xvl={Xvl.shape}")

    print("\nTraining Residual BiLSTM...")
    net = ResidualBiLSTMNet(n_feat=len(RESIDUAL_FEATURES), hidden=HIDDEN)
    net, hist_res = train_residual_net(net, Xtr, Ytr, Xvl, Yvl, device)
    hist_res.to_csv(os.path.join(out_dir, "history_residual.csv"), index=False)
    torch.save(net.state_dict(), os.path.join(out_dir, "residual_model.pt"))

    print("\nEvaluating validation...")
    metrics_df, pred = evaluate(val_cycles, model, Q_du, R_du, net, mu, sig, device)
    summary_df = summarize(metrics_df)

    metrics_df.to_csv(os.path.join(out_dir, "validation_cycle_metrics.csv"), index=False)
    summary_df.to_csv(os.path.join(out_dir, "validation_summary.csv"), index=False)

    print_summary(summary_df, "FAST MODE SUMMARY")

    if pred:
        plot_example(pred[0], os.path.join(out_dir, "example_prediction"))
    plot_summary(summary_df, "FAST MODE RMSE", os.path.join(out_dir, "summary_rmse"))

    return summary_df, metrics_df


def run_lobo(cycles, model, device):
    print("\n" + "█"*80)
    print("FULL MODE | Leave-One-Battery-Out | Current Identified Model")
    print("█"*80)

    out_dir = os.path.join(OUT_DIR, "leave_one_battery_out")
    os.makedirs(out_dir, exist_ok=True)

    all_metrics = []
    fold_summaries = []

    for test_batt, train_all, test_cycles in make_lobo_folds(cycles):
        print("\n" + "─"*80)
        print(f"Test battery: {test_batt}")
        print("─"*80)

        fold_dir = os.path.join(out_dir, f"test_{test_batt}")
        os.makedirs(fold_dir, exist_ok=True)

        fit_cycles, val_cycles = split_train_val(train_all)
        print(f"Fit={len(fit_cycles)} | Val={len(val_cycles)} | Test={len(test_cycles)}")

        Q_du, R_du, hist_du = train_duekf_qr(fit_cycles, val_cycles, model)
        hist_du.to_csv(os.path.join(fold_dir, "history_duekf.csv"), index=False)

        fit_aug = add_du_features(fit_cycles, model, Q_du, R_du)
        val_aug = add_du_features(val_cycles, model, Q_du, R_du)

        Xtr, Ytr, mu, sig = stack_residual_dataset(fit_aug, RESIDUAL_FEATURES, WINDOW)
        Xvl, Yvl, _, _ = stack_residual_dataset(val_aug, RESIDUAL_FEATURES, WINDOW, mu, sig)
        print(f"Residual data: Xtr={Xtr.shape}, Xvl={Xvl.shape}")

        net = ResidualBiLSTMNet(n_feat=len(RESIDUAL_FEATURES), hidden=HIDDEN)
        net, hist_res = train_residual_net(net, Xtr, Ytr, Xvl, Yvl, device)
        hist_res.to_csv(os.path.join(fold_dir, "history_residual.csv"), index=False)
        torch.save(net.state_dict(), os.path.join(fold_dir, "residual_model.pt"))

        metrics_df, pred = evaluate(test_cycles, model, Q_du, R_du, net, mu, sig, device)
        metrics_df["test_battery"] = test_batt
        all_metrics.append(metrics_df)

        fold_summary = summarize(metrics_df)
        fold_summary["test_battery"] = test_batt
        fold_summaries.append(fold_summary)

        metrics_df.to_csv(os.path.join(fold_dir, "test_cycle_metrics.csv"), index=False)
        fold_summary.to_csv(os.path.join(fold_dir, "test_summary.csv"), index=False)

        print_summary(fold_summary, f"FOLD SUMMARY | TEST {test_batt}")

        if pred:
            plot_example(pred[0], os.path.join(fold_dir, "example_prediction"))
        plot_summary(fold_summary, f"RMSE | Test {test_batt}", os.path.join(fold_dir, "summary_rmse"))

    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    fold_summary_df = pd.concat(fold_summaries, ignore_index=True)
    overall = summarize(all_metrics_df)

    all_metrics_df.to_csv(os.path.join(out_dir, "lobo_all_cycle_metrics.csv"), index=False)
    fold_summary_df.to_csv(os.path.join(out_dir, "lobo_fold_summaries.csv"), index=False)
    overall.to_csv(os.path.join(out_dir, "lobo_overall_summary.csv"), index=False)

    print_summary(overall, "LOBO OVERALL SUMMARY")
    plot_summary(overall, "LOBO Overall RMSE", os.path.join(out_dir, "overall_rmse"))

    return overall, all_metrics_df, fold_summary_df


# ============================================================
# 8. Main
# ============================================================

def main():
    print("\n" + "█"*88)
    print(" NASA Training with Current Identified ECM Model")
    print("█"*88)

    set_seed(SEED)
    device = get_device()

    model_json = choose_model_json()
    model = IdentifiedBatteryModel(model_json)

    df = load_clean_dataset()
    cycles = dataframe_to_cycles(df)

    print(f"Device: {device}")
    print(f"FAST_MODE: {FAST_MODE}")
    print(f"Clean CSV: {CLEAN_CSV}")
    print(f"Model JSON: {model_json}")
    print(f"Loaded samples: {len(df)}")
    print(f"Loaded cycles : {len(cycles)}")
    print("Cycles per battery:")
    print(pd.Series([c["battery_id"] for c in cycles]).value_counts().sort_index())

    with open(os.path.join(OUT_DIR, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "FAST_MODE": FAST_MODE,
            "CLEAN_CSV": CLEAN_CSV,
            "MODEL_JSON": model_json,
            "BATTERY_IDS": BATTERY_IDS,
            "FAST_BATTERY_ID": FAST_BATTERY_ID,
            "RESIDUAL_FEATURES": RESIDUAL_FEATURES,
            "DU_EPOCHS": DU_EPOCHS,
            "RES_EPOCHS": RES_EPOCHS,
            "WINDOW": WINDOW,
            "HIDDEN": HIDDEN,
            "RESIDUAL_CLIP": RESIDUAL_CLIP,
            "PRED_CLIP": PRED_CLIP,
        }, f, indent=2)

    if FAST_MODE:
        return run_fast(cycles, model, device)
    else:
        return run_lobo(cycles, model, device)


if __name__ == "__main__":
    main()


