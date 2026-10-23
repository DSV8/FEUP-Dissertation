# CINDERELLA Retrieval Pipelines

This repository contains image retrieval experiments for the CINDERELLA catalogue data. It is intended to be both an experiment codebase and a starting point for further dissertation work. The main comparison is between image-only, collaborative, multimodal, and jointly trained retrieval approaches.

## Start Here

1. Use Python 3.12 or newer. The project is configured for `uv`; install it if needed, then install the locked dependencies:

	 ```bash
	 uv sync
	 ```

	 On Linux, the project selects PyTorch 2.5.1 with CUDA 12.1. On other platforms it selects the CPU PyTorch build. A compatible NVIDIA driver and GPU are needed to use CUDA; training can run on CPU but will be much slower.

2. Make the dataset available locally or on the cluster, then edit `data.dataset_path` in [`configs/common.yaml`](configs/common.yaml). The configured path currently points to a private mounted dataset and will not exist on every machine.

3. Choose a pipeline and run it from the repository root. For example:

	 ```bash
	 uv run python src/main_cbir.py
	 ```

	 The training runners log experiments to Weights & Biases. Authenticate with `wandb login` before a run, or set `WANDB_MODE=offline` to save logs locally for later synchronization.

The `launch_*.sh` files are Slurm job templates that call the corresponding runner with `python3`. Check the partition and QoS directives against the cluster before submission, then submit with `sbatch launch_cbir.sh` (or the matching script). For local use, the `uv run` commands above avoid relying on whichever `python3` happens to be first on `PATH`.

## Choose A Pipeline

| Pipeline | Entry point | Learning signal | What it tests |
| --- | --- | --- | --- |
| CBIR | `src/main_cbir.py` | Query-positive-negative triplets; image embeddings are trained so preferred neighbors are closer | Image-only content-based retrieval |
| CF | `src/main_cf.py` | Query-item relevance targets trained with pointwise MSE | FM, NN, or NCF collaborative scoring using image and tabular features |
| CF-IO | `src/main_cf_io.py` | Same ranked interactions and CF scorer choices, with tabular input disabled | Image-only ablation of the CF approach |
| MM | `src/main_mm.py` | Triplet loss over fused image and tabular embeddings | Whether structured attributes improve retrieval representations |
| Joint | `src/main_joint.py` | CF pointwise loss and MM triplet loss update a shared model | Shared representation learning across both tasks |
| GC-ViT baseline | `src/baseline_gcvit.py` | No task-specific training; evaluates the pretrained image encoder | Reference point for the fine-tuned CBIR model |

For CBIR and MM, smaller query-to-neighbor distances indicate a better match. CF models output relevance scores, where larger scores rank higher. The joint model has both a CF scoring head and an MM embedding path; its two datasets use the same catalogue split.

## Data Contract

The loaders discover catalogue folders with this exact structure:

```text
<dataset-root>/
	Excl_<numeric-id>/
		query_<age>_<height>_<weight>_<cup>_<band>.<image-extension>
		neighbor_<age>_<height>_<weight>_<cup>_<band>_rank_<rank>.<image-extension>
	Fair_<numeric-id>/
		...
```

For example, `query_32_165_60_C_75.jpg` and `neighbor_33_167_61_C_75_rank_1.jpg`. Neighbor ranks define the ground-truth order; rank 1 is best. The `Excl` and `Fair` folder names and numeric IDs are part of the parser contract. The code does not read a separate metadata table.

CF and MM derive five numeric features from each filename: age, height, weight, a cup-size midpoint, and a band-size midpoint. They skip samples when those values cannot be parsed. CBIR only needs valid catalogue folders, query images, and ranked neighbors. If experimenting with missing or differently encoded metadata, update and test the parser and feature conversion together.

The data may contain sensitive personal or clinical images and attributes. Use only approved copies and storage locations; do not commit data, credentials, or generated caches/checkpoints to the repository.

## Configuration

Settings live in [`configs/`](configs/). Every runner loads `common.yaml` and then merges its pipeline-specific YAML over it. Pipeline values override matching common values, including nested keys.

| Configuration | Runner |
| --- | --- |
| `cbir.yaml` | `src/main_cbir.py` |
| `cf.yaml` | `src/main_cf.py` |
| `cf_io.yaml` | `src/main_cf_io.py` |
| `mm.yaml` | `src/main_mm.py` |
| `joint.yaml` | `src/main_joint.py` |
| `baseline_gcvit.yaml` | `src/baseline_gcvit.py` |

`common.yaml` owns the dataset path, catalogue family, train/test ratio, seed, and default cache/checkpoint directories. Relative cache and checkpoint paths resolve from the repository root. Set `data.catalogue_type` to `Excl`, `Fair`, or `Mixed` (the aliases `E`, `F`, and `M` are also accepted). The split is made at catalogue level and stratified by family; in single-family runs, the selected family is filtered after that canonical split. `Mixed` uses both families and includes a catalogue-family feature in the applicable models.

Pipeline YAML files contain their own training hyperparameters, model variants, loader options, and metrics. Change these files to record experiment settings rather than editing runner constants. Check that the image-embedding mode and architecture-specific dimensions remain compatible when changing model configurations.

## Image Processing And Caches

Images are converted to RGB, padded to a square, resized to 224 × 224, converted to tensors, and normalized with ImageNet statistics. The cache stores preprocessed image tensors, not model embeddings, under family-specific directories such as `cache/multimodal/pre_post/Excl/` and `Fair/`. Cache misses can be created on demand. If the preprocessing implementation or source images change, remove or relocate the affected cache before comparing experiments; stale tensors can otherwise make a run inconsistent.

The `training.image_embedding_mode` setting controls whether image features are computed once before training (`precomputed`) or whether the image encoder runs as part of each training/evaluation forward pass (`encoded_at_runtime`). Precomputation can reduce repeated backbone work but consumes memory for the embedding map and prevents gradients from updating the image backbone during that run. Runtime encoding is more flexible for fine-tuning but generally costs more GPU time and memory. The CBIR and MM pipelines support both modes; CF and joint support both through their dataset/model wrappers.

## Training And Evaluation

Catalogue-level train/test splits are shared by triplet construction and ranking evaluation. Do not tune model choices against the test split if it is being used as the final reported result; use a validation split or reserve a separate final test set for dissertation claims.

The code reports pairwise/triplet accuracy and catalogue ranking metrics, including nDCG and cutoff metrics such as precision and recall. CF additionally reports RMSE against normalized rank targets. For comparisons, use the same catalogue family, split ratio, seed, and evaluation settings across runs, and report ranking metrics alongside training loss or accuracy. A model can improve pairwise accuracy without improving the top of the ranked list.

Training runs save `best_model.pt` and `last_model.pt` plus resumable checkpoints under the configured `checkpoints/` directory. Weights & Biases stores experiment history when online logging is enabled. The baseline writes `baseline_gcvit_results.json` at the repository root. Keep large generated artifacts out of version control unless the project explicitly requires them.

## Code Map

- `src/main_*.py`: experiment setup, configuration loading, and wiring of data/model/training components.
- `src/cbir.py`, `src/cf.py`, `src/mm.py`, `src/joint.py`: training loops and public pipeline operations.
- `src/cf_io.py`: image-only CF model construction and dataset adapter.
- `src/models/`: image backbones, feature encoders, fusion layers, collaborative scorers, and the joint model.
- `src/utils/catalogue_utils.py`: the catalogue and filename contract, triplet generation, and canonical splitting.
- `src/utils/{cbir,cf,mm,joint}/`: task-specific dataset, evaluation, and visualization helpers.
- `src/utils/cache_preprocessing.py`: preprocessed image tensor caching.
- `src/configuration.py`: shared YAML loading and recursive override behavior.
- `src/visualize_rank_predicts_*.ipynb`: notebook-based qualitative inspection of model rankings.

## A Practical Extension Path

For a new experiment, start by identifying the supervision signal: a ranked query-item score, a relative triplet preference, or both. Keep catalogue splitting and filename parsing centralized, fit any feature normalization on training data only, and use the same held-out catalogues for all compared models. Put reusable network components in `src/models/`, task-specific data/evaluation code in the matching `src/utils/<task>/` module, and experiment choices in a pipeline YAML file. Add a runner only when the experiment needs a distinct training workflow, then update the pipeline table and code map here.

Before a long cluster run, verify the configured dataset path, catalogue counts, split counts, model variant, embedding mode, cache location, and Weights & Biases mode in the startup output. A short smoke run with a small epoch count is a useful check that data parsing and the full forward/backward path work on the target environment.
