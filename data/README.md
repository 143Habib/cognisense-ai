# Data directory

This directory contains derived research evidence and precomputed tables used by the dashboard.

- CSV/JSON files — precomputed features, out-of-fold predictions, cohort projections, modality metrics, quality summaries, and research evidence.
- `MAUS.zip` — optional local copy of the full original MAUS archive. It is intentionally excluded from Git by `.gitignore`; obtain it through the official dataset record if you have authorized access. Place it directly in this directory to enable raw waveform inspection.
- `maus_research_subset.zip` — optional local curated subset used by the legacy fingertip-PPG utility. It is intentionally excluded from the public repository.

The dashboard can display the saved evidence without either archive. Signal Studio needs an authorized original `MAUS.zip` archive for raw ECG, fingertip PPG, wrist PPG, and GSR/EDA waveform inspection.
