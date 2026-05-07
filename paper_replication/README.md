# Guitar Style Dataset Paper Replication

This folder is a self-contained replication harness for the three experiments in the Guitar Style Dataset paper.

## Quickstart

From `FinalProject/6.S955_project/paper_replication`:

```bash
scripts/run_experiment_1.sh
scripts/run_experiment_2.sh
scripts/run_experiment_3.sh
```

The scripts write raw logs to `logs/`, intermediate files to `work/`, model checkpoints to `pkl/`, and metrics to `results/`.

## Exact Commands Used Locally

```bash
python3.11 -m venv paper_replication/.venv
paper_replication/.venv/bin/python -m pip install --upgrade pip setuptools wheel pyAudioAnalysis==0.3.14 eyed3 pydub imbalanced-learn deep-audio-features==0.2.18
paper_replication/.venv/bin/python -m pip install torch==2.9.0 librosa==0.11.0 pandas==2.3.0 matplotlib==3.10.1 tqdm==4.67.1 soundfile==0.13.1 pyyaml==6.0.2 tabulate==0.9.0
cd paper_replication
.venv/bin/python scripts/verify_inputs.py --config configs/experiment_1.yaml
.venv/bin/python scripts/verify_inputs.py --config configs/experiment_2.yaml
.venv/bin/python scripts/verify_inputs.py --config configs/experiment_3.yaml
.venv/bin/python scripts/run_svm_original.py --config configs/experiment_1.yaml
.venv/bin/python scripts/run_svm_original.py --config configs/experiment_2.yaml
.venv/bin/python scripts/run_svm_original.py --config configs/experiment_3.yaml
bash -lc 'source scripts/common_env.sh && cd "$REPLICATION_ROOT" && "$PYTHON_BIN" scripts/run_cnn_original.py --config configs/experiment_1.yaml'
.venv/bin/python scripts/collect_results.py
.venv/bin/python scripts/write_replication_report.py
```

The CNN command above completed experiment 1 fold 0 and was stopped during fold 1 because full CNN reproduction on CPU exceeded the interactive execution budget. The full-fidelity rerun commands remain the `scripts/run_experiment_*.sh` commands in Quickstart.

## Docker

Docker is preferred for reproducibility, and `environment/Dockerfile` is provided. The execution host did not have a `docker` binary, so the local run used `.venv`.

If Docker is available:

```bash
cd paper_replication
docker compose -f environment/docker-compose.yml build
docker compose -f environment/docker-compose.yml run --rm replication scripts/run_experiment_1.sh
docker compose -f environment/docker-compose.yml run --rm replication scripts/run_experiment_2.sh
docker compose -f environment/docker-compose.yml run --rm replication scripts/run_experiment_3.sh
```

## Data Location

The harness reuses the existing local dataset and does not redownload it:

- Dataset root: `../guitar_style_dataset/magcil-guitar_style_dataset-eb27d7b/data`
- WAV root: `../guitar_style_dataset/magcil-guitar_style_dataset-eb27d7b/data/wav`
- Archive: `../guitar_style_dataset-v1.0.0.zip`
- Paper PDF used: `../../Guitar_dataset.pdf`

The user-provided PDF path inside `6.S955_project` was absent; the existing local PDF one directory above was used.

## Expected CPU Runtime

SVM takes seconds to minutes after feature caching. CNN is full-fidelity and CPU-only; experiment 1 fold 0 trained to epoch 58 before early stopping, so all 11 CNN folds should be treated as an overnight-class run on this host rather than an interactive job. Each CNN rerun uses the original 500-epoch budget and package early stopping.

## Outputs

- `results/experiment_*/svm/summary.json`
- `results/experiment_*/cnn/summary.json`
- `results/comparison_table.csv`
- `results/comparison_table.json`
- `results/artifact_manifest.txt`
- `replication_report.md`

