# Artifact description

IndexWise-Recsys is a CPU-only research artifact for calibrated ANN index selection in recommender systems. The canonical workflow and scientific invariants are defined in `README.md`, `reproduction.md`, and `critical_experiment_contract.md`.

The artifact supports four datasets, five CPU FAISS methods, and U2I/I2I evaluation. It produces main evidence under `results/main/`, analyses under `results/analyses/`, paper tables and figures under `results/paper/`, and metadata/validation under `results/_meta/`.

Install `requirements-cpu.txt` for every canonical result. `requirements-optional.txt` contains only optional CPU embedding and ANN backends. No dataset is bundled or downloaded by the canonical workflow.

IndexWise-Recsys is evaluated as a CPU-only framework. GPU acceleration and GPU-specific latency behavior are outside the current scope.
