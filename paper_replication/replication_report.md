# Replication Report

Generated from `paper_replication/results`.

## Scope

This replication attempts the three paper experiments using the local original `guitar_style_dataset` repository and the already-downloaded dataset. Each experiment runs **SVM**, **mel-spectrogram CNN**, and the notebook-aligned **MFCC CNN** (`mfcc_cnn`, Alar-style architecture on guitar audio) on the **same official splits** (Table 3 baselines for SVM and mel CNN). The only MFCC variant reported is `mfcc_cnn`; non-aligned baselines and `n_fft` sweeps are not included. MFCC summaries use the guitar-paper **CNN** targets in `paper_target` for delta columns so Stage 3 is comparable to the published mel-CNN numbers, not a separate paper table.

## Environment Outcome

- Docker is the preferred reproducibility target, and a Dockerfile is provided in `environment/`.
- Docker was not installed on the execution host, so runs used the local `.venv` fallback.
- The local fallback uses Python 3.11 and modern Torch because the original CNN pins include `torch==1.9.0`, which is not compatible with the available host Python.
- Compatibility shims were limited to runtime issues: ignore the removed `verbose` keyword in `ReduceLROnPlateau`, skip a diagnostic histogram plot that crashed in headless macOS execution, and force Torch dataloaders to `num_workers=0` after shared-memory worker launch failed.

## Results

When available, `comparison_table.csv` includes t-based 95% CI columns for reproduced MFCC metrics; the compact table below reports means and deltas.

| Experiment | Method | Status | Folds | Paper Acc | Reproduced Acc | Acc Delta | Paper F1 | Reproduced F1 | F1 Delta | Within Paper F1 Std |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| experiment_1 | svm | completed | 5/5 | 67.80 | 64.44 | -3.36 | 62.30 | 59.57 | -2.73 | True |
| experiment_1 | cnn | completed | 5/5 | 76.50 | 74.07 | -2.43 | 76.90 | 75.37 | -1.53 | True |
| experiment_1 | mfcc_cnn | completed | 5/5 | 76.50 | 94.89 | 18.39 | 76.90 | 94.78 | 17.88 | False |
| experiment_2 | svm | completed | 3/3 | 84.20 | 82.33 | -1.87 | 81.20 | 81.75 | 0.55 | True |
| experiment_2 | cnn | completed | 3/3 | 81.10 | 82.33 | 1.23 | 83.10 | 84.68 | 1.58 | False |
| experiment_2 | mfcc_cnn | completed | 3/3 | 81.10 | 80.22 | -0.88 | 83.10 | 80.19 | -2.91 | False |
| experiment_3 | svm | completed | 3/3 | 83.40 | 84.88 | 1.48 | 81.90 | 85.29 | 3.39 | True |
| experiment_3 | cnn | completed | 3/3 | 80.70 | 78.14 | -2.56 | 82.30 | 80.38 | -1.92 | True |
| experiment_3 | mfcc_cnn | completed | 3/3 | 80.70 | 75.23 | -5.47 | 82.30 | 77.42 | -4.88 | False |

Overall verdict: **partially replicated**.

(Verdict expects **9** completed rows when MFCC summaries are present in the table, otherwise **6**.)

## Matched Exactly

- Local dataset archive is reused, not redownloaded.
- Official split JSON files are used for the three paper experiments.
- SVM grid values match the paper and original repo.
- CNN architecture and training defaults come from `deep-audio-features==0.2.18`, the original repo dependency.

## Approximations And Deviations

- The paper PDF path requested by the user was not present inside `6.S955_project`; the actual local PDF used is `../../Guitar_dataset.pdf` from `paper_replication`.
- Docker execution was not possible because the host has no `docker` binary.
- Local Python 3.11 required Torch 2.9 rather than the original CNN requirement `torch==1.9.0`.
- The original SVM code path extracts one long-term averaged feature vector per WAV file, while the paper prose describes 1-second segment-level vectors.
- CNN F1 uses the original wrapper formula, the harmonic mean of macro precision and macro recall; sklearn macro-F1 is also saved in fold metrics.
- CNN full reproduction was blocked by CPU runtime in this interactive session: experiment 1 fold 0 completed, fold 1 was interrupted at epoch 15, and experiments 2-3 were left as full-fidelity rerun commands.

## Confidence

Confidence is highest for data/split parity, SVM code mapping, and the completed SVM metrics. Confidence is lower for exact CNN numerical reproduction because the host could not run the original Docker-like Python/Torch stack and the full 11-fold CPU run exceeded the interactive budget.
