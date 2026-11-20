from __future__ import annotations
import json, mimetypes, socket, threading, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote
import numpy as np
import pandas as pd
from src.multimodal_core import ROOT, get_raw_trial, MODALITIES, FS

HOST = "0.0.0.0"
PORT = 8501
STATIC = ROOT / "static"
DATA = ROOT / "data"

print("[CogniSense] Loading the multimodal research workspace...")
required = [
    DATA / "maus_multimodal_trial_features.csv",
    DATA / "multimodal_loso_predictions.csv", DATA / "multimodal_pca.csv",
    DATA / "research_evidence.json", DATA / "modality_metrics_fusion.json",
]
missing = [str(p.name) for p in required if not p.exists()]
if missing:
    raise FileNotFoundError("Required research assets are missing: " + ", ".join(missing))

FEATURES = pd.read_csv(DATA / "maus_multimodal_trial_features.csv", dtype={"subject": str})
OOF = pd.read_csv(DATA / "multimodal_loso_predictions.csv", dtype={"subject": str})
PCA = pd.read_csv(DATA / "multimodal_pca.csv", dtype={"subject": str})
IMPORTANCE = pd.read_csv(DATA / "multimodal_feature_importance.csv")
NASA = pd.read_csv(DATA / "nasa_tlx_by_trial.csv", dtype={"subject": str})
PERFORMANCE = pd.read_csv(DATA / "task_response_log_summary.csv", dtype={"subject": str})
QUALITY = pd.read_csv(DATA / "signal_quality_by_trial.csv", dtype={"subject": str})
PARTICIPANTS = pd.read_csv(DATA / "participant_context.csv", dtype={"subject": str})
EVIDENCE = json.loads((DATA / "research_evidence.json").read_text(encoding="utf-8"))
MODEL_COMPARISON = []
for mod in ["ecg", "fppg", "wppg", "gsr", "ibi", "fusion"]:
    p = DATA / f"modality_metrics_{mod}.json"
    if p.exists(): MODEL_COMPARISON.append(json.loads(p.read_text(encoding="utf-8")))
PPG_REFERENCE = EVIDENCE["model_reference"]["fingertip_ppg_baseline"]
MULTICLASS = json.loads((DATA / "multiclass_fusion_metrics.json").read_text(encoding="utf-8"))
SUBJECTS = sorted(FEATURES["subject"].astype(str).unique())
TASK_ORDER = [0, 2, 3]
TASK_LABELS = {0: "0-back", 2: "2-back", 3: "3-back"}


def clean_records(df: pd.DataFrame):
    clean = df.replace({np.nan: None, np.inf: None, -np.inf: None})
    return json.loads(clean.to_json(orient="records"))


def _summary():
    nasa = NASA.groupby("task_n")["nasa_tlx_adjusted"].agg(["mean", "std", "count"]).reset_index()
    phys_cols = {
        "ecg_hr": "ecg_hr_mean_bpm_mean", "ecg_sdnn": "ecg_hrv_sd_ms_mean",
        "fppg_hr": "fppg_hr_mean_bpm_mean", "fppg_sdnn": "fppg_hrv_sd_ms_mean",
        "wppg_hr": "wppg_hr_mean_bpm_mean", "wppg_sdnn": "wppg_hrv_sd_ms_mean",
        "gsr_tonic": "gsr_tonic_mean_mean", "gsr_phasic": "gsr_phasic_sd_mean",
    }
    physiology = []
    for task in TASK_ORDER:
        part = FEATURES[FEATURES.task_n.astype(int) == task]
        row = {"task_n": task, "task": TASK_LABELS[task], "count": int(len(part))}
        for key, col in phys_cols.items():
            row[key] = float(part[col].mean()) if col in part else None
        physiology.append(row)
    nasa_rows = [{"task_n": int(r.task_n), "task": TASK_LABELS[int(r.task_n)],
                  "mean": float(r["mean"]), "std": float(r["std"]) if pd.notna(r["std"]) else None,
                  "count": int(r["count"])} for _, r in nasa.iterrows()]
    perf = PERFORMANCE.groupby("task_n").agg(
        mean_rt_ms=("mean_response_time_ms", "mean"), median_rt_ms=("median_response_time_ms", "median"),
        positive_rt_records=("positive_rt_records", "mean")).reset_index()
    perf_rows = [{"task_n": int(r.task_n), "task": TASK_LABELS[int(r.task_n)],
                  "mean_rt_ms": float(r.mean_rt_ms) if pd.notna(r.mean_rt_ms) else None,
                  "median_rt_ms": float(r.median_rt_ms) if pd.notna(r.median_rt_ms) else None,
                  "positive_rt_records": float(r.positive_rt_records) if pd.notna(r.positive_rt_records) else None}
                 for _, r in perf.iterrows()]
    task_counts = {TASK_LABELS[k]: int((FEATURES.task_n.astype(int) == k).sum()) for k in TASK_ORDER}
    quality = EVIDENCE["signal_quality"]
    return {
        "project": "CogniSense AI",
        "subtitle": "Multimodal Cognitive Workload Observatory",
        "dataset": EVIDENCE["dataset"],
        "participants": len(SUBJECTS), "trials": len(FEATURES),
        "subjects": SUBJECTS, "task_counts": task_counts,
        "modalities": MODALITIES, "sampling_rates_hz": FS,
        "signal_quality": quality, "nasa_by_task": nasa_rows,
        "physiology_by_task": physiology, "response_log_by_task": perf_rows,
        "model_comparison": MODEL_COMPARISON,
        "ppg_reference": PPG_REFERENCE,
        "multimodal_binary": next(x["binary"] for x in MODEL_COMPARISON if x["modality"] == "fusion"),
        "multimodal_multiclass": MULTICLASS,
        "statistical_evidence": EVIDENCE["statistical_evidence"],
        "ibi_sensor_agreement": EVIDENCE["ibi_sensor_agreement"],
        "ibi_agreement_points": clean_records(FEATURES[["subject","trial","ibi_ecg_mean_ms","ibi_fppg_mean_ms","ibi_wppg_mean_ms"]]),
        "limitations": EVIDENCE["limitations"],
    }

SUMMARY = _summary()
SUMMARY["raw_dataset_available"] = (DATA / "MAUS.zip").is_file()


class Handler(BaseHTTPRequestHandler):
    server_version = "CogniSenseResearch/2.0"

    def log_message(self, fmt, *args):
        pass

    def _send(self, body: bytes, status=200, content_type="application/json; charset=utf-8", headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if headers:
            for k, v in headers.items(): self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send(body, status=status)

    def _static(self, relpath):
        relpath = unquote(relpath).lstrip("/")
        path = (STATIC / relpath).resolve()
        if not str(path).startswith(str(STATIC.resolve())) or not path.is_file():
            self._send(b"Not found", 404, "text/plain; charset=utf-8"); return
        mime, _ = mimetypes.guess_type(str(path))
        self._send(path.read_bytes(), 200, mime or "application/octet-stream")

    def do_GET(self):
        parsed = urlparse(self.path); path = parsed.path; qs = parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                self._send((STATIC / "index.html").read_bytes(), 200, "text/html; charset=utf-8")
            elif path.startswith("/static/"):
                self._static(path[len("/static/"):])
            elif path == "/api/health":
                self._json({"ok": True, "app": "CogniSense AI", "version": "2.0-multimodal", "raw_dataset_available": (DATA / "MAUS.zip").is_file()})
            elif path == "/api/summary":
                self._json(SUMMARY)
            elif path == "/api/trials":
                self._json(clean_records(OOF))
            elif path == "/api/pca":
                self._json(clean_records(PCA))
            elif path == "/api/importance":
                self._json(clean_records(IMPORTANCE.head(24)))
            elif path == "/api/signal":
                subject = str(qs.get("subject", [SUBJECTS[0]])[0]).zfill(3)
                trial = int(qs.get("trial", [1])[0]); window = int(qs.get("window", [1])[0])
                if subject not in SUBJECTS:
                    self._json({"error": "Unknown participant."}, 404); return
                if not (DATA / "MAUS.zip").is_file():
                    self._json({"error": "The original MAUS.zip archive is not installed. Place an authorized copy at data/MAUS.zip to enable raw waveform inspection."}, 424); return
                payload = get_raw_trial(subject, trial, window, seconds=20)
                row = FEATURES[(FEATURES.subject.astype(str) == subject) & (FEATURES.trial.astype(int) == trial)].iloc[0]
                pred = OOF[(OOF.subject.astype(str) == subject) & (OOF.trial.astype(int) == trial)].iloc[0]
                payload["context"] = {
                    "nasa_tlx_adjusted": float(row.nasa_tlx) if pd.notna(row.nasa_tlx) else None,
                    "psqi": float(row.psqi) if pd.notna(row.psqi) else None,
                    "task_condition": str(row.task_condition),
                    "fusion_prediction": str(pred.predicted_group),
                    "fusion_score": float(pred.oof_score) if pd.notna(pred.oof_score) else None,
                    "prediction_correct": bool(pred.correct),
                }
                self._json(payload)
            elif path == "/api/participant":
                subject = str(qs.get("subject", [SUBJECTS[0]])[0]).zfill(3)
                if subject not in SUBJECTS:
                    self._json({"error": "Unknown participant."}, 404); return
                trials = FEATURES[FEATURES.subject.astype(str) == subject].merge(
                    OOF[["subject", "trial", "predicted_group", "oof_score", "correct"]],
                    on=["subject", "trial"], how="left")
                trials = trials.merge(NASA[["subject", "trial", "nasa_tlx_adjusted"]], on=["subject", "trial"], how="left")
                self._json({"subject": subject, "psqi": float(PARTICIPANTS[PARTICIPANTS.subject.astype(str)==subject].psqi.iloc[0]),
                            "trials": clean_records(trials[["subject","trial","task_condition","nasa_tlx_adjusted","predicted_group","oof_score","correct"]])})
            elif path == "/api/download":
                name = qs.get("name", [""])[0]
                allowed = {
                    "research_evidence.json": DATA / "research_evidence.json",
                    "multimodal_trial_features.csv": DATA / "maus_multimodal_trial_features.csv",
                    "multimodal_loso_predictions.csv": DATA / "multimodal_loso_predictions.csv",
                    "ppg_reference_loso_predictions.csv": DATA / "loso_oof_predictions.csv",
                    "nasa_tlx_by_trial.csv": DATA / "nasa_tlx_by_trial.csv",
                    "subject_task_level_evidence.csv": DATA / "subject_task_level_evidence.csv",
                    "task_response_log_summary.csv": DATA / "task_response_log_summary.csv",
                    "signal_quality_by_trial.csv": DATA / "signal_quality_by_trial.csv",
                    "multimodal_trial_features.csv": DATA / "maus_multimodal_trial_features.csv",
                    "multimodal_loso_predictions.csv": DATA / "multimodal_loso_predictions.csv",
                    "sensor_comparison.csv": DATA / "sensor_comparison_summary.csv",
                }
                file = allowed.get(name)
                if not file or not file.is_file(): self._send(b"Not found", 404, "text/plain; charset=utf-8"); return
                mime, _ = mimetypes.guess_type(str(file));headers={"Content-Disposition":f'attachment; filename="{file.name}"'}
                self._send(file.read_bytes(), 200, mime or "application/octet-stream", headers)
            else:
                self._send(b"Not found", 404, "text/plain; charset=utf-8")
        except Exception as exc:
            self._json({"error": str(exc)}, 500)


def _lan_ip():
    try:
        s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.connect(("8.8.8.8",80));ip=s.getsockname()[0];s.close()
        return ip
    except Exception:
        try:return socket.gethostbyname(socket.gethostname())
        except Exception:return ""


def main():
    httpd=ThreadingHTTPServer((HOST,PORT),Handler)
    local=f"http://127.0.0.1:{PORT}";lan=_lan_ip()
    print("\n"+"─"*66)
    print("  COGNISENSE AI  /  MULTIMODAL RESEARCH OBSERVATORY")
    print("─"*66)
    print(f"  Laptop / desktop : {local}")
    if lan and not lan.startswith("127."):
        print(f"  Phone on same Wi-Fi: http://{lan}:{PORT}")
    print("\n  Keep this terminal open while using the website.")
    print("  Press CTRL+C here to stop the local server.")
    print("  LAN access is for your trusted private Wi-Fi only.\n")
    threading.Timer(1.0,lambda:webbrowser.open(local)).start()
    try:httpd.serve_forever()
    except KeyboardInterrupt:print("\n[CogniSense] Server stopped.")
    finally:httpd.server_close()

if __name__=="__main__":main()
