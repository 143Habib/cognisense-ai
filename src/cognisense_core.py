from __future__ import annotations
import json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks, welch
from scipy.integrate import trapezoid
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SUBSET_ZIP = DATA_DIR / "maus_research_subset.zip"
EXTRACT_DIR = DATA_DIR / "maus_subset"
FEATURES_CSV = DATA_DIR / "maus_trial_features.csv"
OOF_CSV = DATA_DIR / "loso_oof_predictions.csv"
REPORT_JSON = DATA_DIR / "model_report.json"
FS_DEFAULT = 256
TRIAL_SECONDS = 300
WINDOW_SECONDS = 50
WINDOWS_PER_TRIAL = 6


def ensure_dataset_extracted() -> Path:
    """Extract the curated MAUS fingertip-PPG + NASA-TLX subset once."""
    marker = EXTRACT_DIR / "MAUS" / "Data" / "Raw_data" / "002" / "inf_ppg.csv"
    if not marker.exists():
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        if not SUBSET_ZIP.exists():
            raise FileNotFoundError(f"Missing curated dataset archive: {SUBSET_ZIP}")
        with zipfile.ZipFile(SUBSET_ZIP, "r") as zf:
            zf.extractall(EXTRACT_DIR)
    return EXTRACT_DIR / "MAUS"


def load_features() -> pd.DataFrame:
    return pd.read_csv(FEATURES_CSV, dtype={"subject": str})


def load_oof() -> pd.DataFrame:
    return pd.read_csv(OOF_CSV, dtype={"subject": str})


def load_report() -> dict:
    with open(REPORT_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ppg_trial(subject: str, trial_number: int) -> np.ndarray:
    maus_root = ensure_dataset_extracted()
    subject_id = str(subject).zfill(3)
    path = maus_root / "Data" / "Raw_data" / subject_id / "inf_ppg.csv"
    df = pd.read_csv(path)
    idx = int(trial_number) - 1
    if idx < 0 or idx >= df.shape[1]:
        raise ValueError("Trial number must be between 1 and 6")
    return pd.to_numeric(df.iloc[:, idx], errors="coerce").interpolate(limit_direction="both").to_numpy(dtype=float)


def _safe_skew(x: np.ndarray) -> float:
    return float(skew(x, bias=False)) if np.std(x) > 1e-9 else 0.0


def _safe_kurtosis(x: np.ndarray) -> float:
    return float(kurtosis(x, bias=False)) if np.std(x) > 1e-9 else 0.0


def ppg_window_features(signal: np.ndarray, fs: int = FS_DEFAULT) -> dict:
    """Same deterministic fingertip-PPG feature routine used for the MAUS feature cache."""
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < fs * 10:
        raise ValueError("Each PPG window must contain at least 10 seconds of valid samples.")

    sos = butter(3, [0.5, 8.0], btype="bandpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, x)
    med = np.median(y)
    mad = np.median(np.abs(y - med)) * 1.4826 + 1e-8
    z = np.clip((y - med) / mad, -10, 10)

    peaks, props = find_peaks(
        z,
        distance=int(fs * 0.33),
        prominence=max(0.25, float(np.std(z)) * 0.25),
    )
    ibi = np.diff(peaks) / fs if len(peaks) > 1 else np.array([], dtype=float)
    ibi = ibi[(ibi >= 0.33) & (ibi <= 1.5)]

    amplitudes = []
    for peak in peaks:
        start = max(0, peak - int(fs * 0.35))
        trough = start + int(np.argmin(z[start : peak + 1]))
        amplitudes.append(z[peak] - z[trough])
    amplitudes = np.asarray(amplitudes, dtype=float)

    def band_power(low: float, high: float) -> float:
        f, p = welch(z, fs=fs, nperseg=min(len(z), fs * 8))
        mask = (f >= low) & (f < high)
        return float(trapezoid(p[mask], f[mask])) if np.sum(mask) > 1 else 0.0

    if len(ibi):
        hr = float(60.0 / np.median(ibi))
        sdnn = float(np.std(ibi, ddof=1) * 1000) if len(ibi) > 1 else 0.0
        rmssd = float(np.sqrt(np.mean(np.diff(ibi) ** 2)) * 1000) if len(ibi) > 2 else 0.0
        pnn50 = float(np.mean(np.abs(np.diff(ibi)) * 1000 > 50)) if len(ibi) > 2 else 0.0
        ibi_mean = float(np.mean(ibi) * 1000)
    else:
        hr = sdnn = rmssd = pnn50 = ibi_mean = 0.0

    return {
        "mean": float(np.mean(z)),
        "std": float(np.std(z)),
        "iqr": float(np.percentile(z, 75) - np.percentile(z, 25)),
        "range": float(np.percentile(z, 99) - np.percentile(z, 1)),
        "skew": _safe_skew(z),
        "kurtosis": _safe_kurtosis(z),
        "d1_abs": float(np.mean(np.abs(np.diff(z)))),
        "d2_abs": float(np.mean(np.abs(np.diff(z, n=2)))),
        "slope": float(np.polyfit(np.arange(len(z)), z, 1)[0]),
        "peak_rate": float(len(peaks) / (len(z) / fs) * 60),
        "peak_amp_median": float(np.median(amplitudes)) if len(amplitudes) else 0.0,
        "peak_amp_sd": float(np.std(amplitudes)) if len(amplitudes) else 0.0,
        "power_low": band_power(0.5, 1.0),
        "power_mid": band_power(1.0, 2.0),
        "power_high": band_power(2.0, 4.0),
        "hr_median": hr,
        "ibi_mean_ms": ibi_mean,
        "ibi_sdnn_ms": sdnn,
        "ibi_rmssd_ms": rmssd,
        "ibi_pnn50": pnn50,
        "ibi_count": int(len(ibi)),
    }


def trial_features_from_signal(signal: np.ndarray, fs: int = FS_DEFAULT) -> pd.DataFrame:
    """Create one-row, 84-feature DataFrame from a five-minute PPG trial."""
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]
    required = int(fs * TRIAL_SECONDS)
    if len(x) < required:
        raise ValueError(
            f"Need at least {TRIAL_SECONDS} seconds at {fs} Hz ({required:,} samples). "
            f"Received {len(x):,} valid samples."
        )
    x = x[:required]
    window_rows = []
    samples_per_window = int(fs * WINDOW_SECONDS)
    for i in range(WINDOWS_PER_TRIAL):
        segment = x[i * samples_per_window : (i + 1) * samples_per_window]
        window_rows.append(ppg_window_features(segment, fs=fs))
    w = pd.DataFrame(window_rows)
    row = {}
    for feature in w.columns:
        values = w[feature].to_numpy(dtype=float)
        row[f"ppg_{feature}_mean"] = float(np.nanmean(values))
        row[f"ppg_{feature}_std"] = float(np.nanstd(values))
        row[f"ppg_{feature}_slope"] = float(np.polyfit(np.arange(6), values, 1)[0])
        row[f"ppg_{feature}_delta"] = float(values[-1] - values[0])
    return pd.DataFrame([row])


def train_final_model(features: pd.DataFrame | None = None):
    """Train the deployment/demo model on all MAUS trials; not an independent validation."""
    from sklearn.ensemble import RandomForestClassifier
    if features is None:
        features = load_features()
    feature_cols = [c for c in features.columns if c.startswith("ppg_") and c not in {"ppg_mean_hr_bpm"}]
    X = features[feature_cols]
    y = (features["task_n"].astype(int) != 0).astype(int)
    model = RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=2,
        max_features=0.5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model, feature_cols


def get_trial_metadata(features: pd.DataFrame, subject: str, trial: int) -> pd.Series:
    match = features[(features["subject"].astype(str) == str(subject)) & (features["trial"].astype(int) == int(trial))]
    if match.empty:
        raise KeyError(f"No cached trial found for subject {subject}, trial {trial}")
    return match.iloc[0]
