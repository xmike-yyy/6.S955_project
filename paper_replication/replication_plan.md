# Replication Plan

## Paper Experiments

The paper defines three split protocols over the same 549-file Guitar Style Dataset. Table 3 reports SVM and CNN results for each split, so this harness runs both model families for each experiment.


| Experiment   | Split JSON        | Task                         | Train/Test Size             | Paper SVM                    | Paper CNN                   |
| ------------ | ----------------- | ---------------------------- | --------------------------- | ---------------------------- | --------------------------- |
| Experiment 1 | `folds.json`      | generalize across exercises  | 441 / 108 per fold, 5 folds | Acc 67.8, F1 62.3 (std 11.6) | Acc 76.5, F1 76.9 (std 8.0) |
| Experiment 2 | `guitars.json`    | generalize across guitars    | 366 / 183 per fold, 3 folds | Acc 84.2, F1 81.2 (std 2.0)  | Acc 81.1, F1 83.1 (std 1.2) |
| Experiment 3 | `amplifiers.json` | generalize across amplifiers | 366 / 183 per fold, 3 folds | Acc 83.4, F1 81.9 (std 5.0)  | Acc 80.7, F1 82.3 (std 3.7) |


## Paper Preprocessing And Models

- Audio only, WAV files resampled to 8 kHz.
- SVM: paper prose says 1-second segments, pyAudioAnalysis 136-D features, RBF SVM, grid search over `C=[0.1,1,10,50,100,1000]` and `gamma=[0.0001,0.001,0.01,0.1,1,10]`.
- CNN: 1-second segments, mel-spectrogram, 50 ms STFT window with no overlap, 128 mel bins, 20 x 128 input, 4 conv layers, 5 x 5 kernels, channels 32 to 256, BatchNorm, LeakyReLU, MaxPool, dense layers 2048 to 1048 to 256 to 9.

## Code Mapping


| Paper item             | Local code path                                                     | Harness mapping                                             |
| ---------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------- |
| Official split files   | `guitar_style_dataset/.../data/*.json`                              | `configs/experiment_*.yaml`                                 |
| SVM feature extraction | `utils/feature_extraction.py` and pyAudioAnalysis                   | `scripts/run_svm_original.py`                               |
| SVM grid search        | `utils/utils.py::custom_folds_train`                                | `scripts/run_svm_original.py` with machine-readable outputs |
| CNN data segmentation  | `deep_audio_features_wrapper/deep_audio_utils.py::prepare_dirs`     | `scripts/run_cnn_original.py`                               |
| CNN training           | `deep_audio_features.bin.basic_training.train_model`                | `scripts/run_cnn_original.py`                               |
| CNN evaluation         | `deep_audio_features_wrapper/deep_audio_utils.py::validate_on_test` | `scripts/run_cnn_original.py`                               |


## Assumptions

- The original repo code is treated as the executable source of truth when it conflicts with incomplete paper details.
- The SVM implementation is deliberately file-level because pyAudioAnalysis `directory_feature_extraction` in the original dependency long-term averages each WAV into one vector. This differs from the paper's segment-level prose and is documented as a methodological divergence.
- Random seed is set to 0 wherever the stack exposes a seed. The original CNN package also hard-codes Torch seed 0.
- CPU-only execution is used.
- Docker is preferred but unavailable on the host, so local venv execution is the active fallback.

## Execution Outcome

- All three SVM experiments completed with the original split JSON files.
- CNN experiment 1 was attempted full-fidelity; fold 0 completed and fold 1 was interrupted at epoch 15 after runtime was classified as an interactive CPU-only blocker.
- CNN experiments 2 and 3 are configured and runnable, but were not launched after experiment 1 showed the full 11-fold CNN run would be an overnight-class job on the available CPU-only host.