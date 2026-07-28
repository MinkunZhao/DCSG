# DCSG

## Dual Channel Learning Enhanced with Semantic Graph for Recommendation

This repository contains the implementation of **DCSG** (*Dual Channel Learning Enhanced with Semantic Graph for Recommendation*), a graph-based recommender system that combines collaborative interaction signals with LLM-derived semantic representations of users and items.

## Overview

Collaborative filtering is effective when interaction data are dense, but it can struggle to express semantic relations that are absent from the observed user-item graph. DCSG introduces a semantic channel alongside the collaborative channel:

1. **Collaborative channel.** A graph recommender propagates trainable user and item embeddings over the normalized user-item interaction graph.
2. **Semantic channel.** LLM-generated user and item profiles are encoded as dense embeddings. DCSG builds a weighted user-item semantic graph by retaining each user's top-`k` most similar items, then applies graph attention to the projected profile embeddings.
3. **Adaptive fusion.** A feature-wise sigmoid gate learns to combine collaborative and semantic representations for every user and item.
4. **Joint optimization.** The default DCSG variants optimize BPR ranking loss together with embedding regularization, semantic knowledge distillation, and contrastive consistency. The gate can additionally be supervised by cached LLM-generated labels.

The main configuration is `lightgcn_dcsg`, with LightGCN as the collaborative backbone. DCSG variants are also provided for SGL, SimGCL, GCCF, GraphAU, NCL, SGCF, DirectAU, AdaGCL, and AutoCF where matching files and configurations are present.

## Repository Layout

```text
DCSG/
+-- encoder/
|   +-- train_encoder.py            # Training entry point
|   +-- config/
|   |   +-- configurator.py         # Command-line and YAML configuration loader
|   |   +-- modelconf/              # Baseline and DCSG YAML configurations
|   +-- data_utils/                 # Interaction loading and graph construction
|   +-- models/
|   |   +-- general_cf/             # CF backbones and *_dcsg variants
|   |   +-- base_model.py
|   |   +-- loss_utils.py
|   +-- trainer/                    # Optimization, early stopping, metrics, logging
|   +-- scripts/                    # Experiment automation utilities
+-- data/
|   +-- amazon/                     # Amazon interactions, profiles, and embeddings
|   +-- yelp/                       # Yelp interactions, profiles, and embeddings
|   +-- steam/                      # Steam interactions, profiles, and embeddings
+-- generation/
|   +-- user/                       # User-profile prompting example
|   +-- item/                       # Item-profile prompting example
|   +-- emb/                        # Profile-embedding example
|   +-- instruction/                # Dataset-specific prompt instructions
```

## Environment

The DCSG implementation uses CUDA tensors internally; run the main experiments in a CUDA-enabled PyTorch environment.

Recommended setup:

```bash
conda create -n dcsg python=3.10 -y
conda activate dcsg

# Install the PyTorch build matching the local CUDA version first.
pip install torch

pip install numpy scipy scikit-learn pyyaml tqdm requests openai matplotlib
pip install torch-geometric torch-sparse
```

`torch-geometric` and `torch-sparse` must match the installed PyTorch and CUDA versions. Use the PyTorch Geometric installation matrix if `pip` does not provide a compatible wheel.

The optional profile-generation scripts use the legacy `openai` Python client API. They are separate from the default training path and may require a compatible client version or a small API migration for newer SDK releases.

## Data Format

The supported training datasets are `amazon`, `yelp`, and `steam`. The data loader expects the following files under `data/{dataset}/`:

```text
data/{dataset}/
+-- trn_mat.pkl        # Sparse user-item training interaction matrix
+-- val_mat.pkl        # Sparse validation interaction matrix
+-- tst_mat.pkl        # Sparse test interaction matrix
+-- usr_prf.json       # User profiles, keyed by user ID
+-- itm_prf.json       # Item profiles, keyed by item ID
+-- usr_emb_np.pkl     # User profile embeddings
+-- itm_emb_np.pkl     # Item profile embeddings
```

The three interaction matrices must share the shape `[num_users, num_items]`, which defines the user and item ID space. Profile embedding rows must align with these IDs. The semantic projection layer expects **1536-dimensional** profile embeddings.

The repository includes processed artifacts for all supported datasets. To prepare new data, create the same aligned interaction, profile, and embedding files and add a dataset-specific block to the selected YAML configuration. The current data handler accepts only Amazon, Yelp, and Steam.

### Semantic Graph Construction

At load time, DCSG L2-normalizes profile embeddings, computes user-item cosine similarities, and retains the `sem_graph_topk` highest-scoring items per user as weighted semantic-graph edges. `sem_graph_topk` is configured per dataset in `encoder/config/modelconf/*_dcsg.yml` and is `10` in the supplied main configurations.

## Training

Run commands from the repository root. The base command is:

```bash
python encoder/train_encoder.py \
  --model lightgcn_dcsg \
  --dataset amazon \
  --device cuda \
  --cuda 0 \
  --seed 2025
```

`--model` selects `encoder/config/modelconf/{model}.yml` and the matching implementation in `encoder/models/general_cf/`. For example:

```bash
# Main DCSG model on Yelp
python encoder/train_encoder.py --model lightgcn_dcsg --dataset yelp --device cuda --cuda 0

# DCSG with self-supervised graph learning backbone
python encoder/train_encoder.py --model sgl_dcsg --dataset steam --device cuda --cuda 0

# DCSG with SimGCL backbone
python encoder/train_encoder.py --model simgcl_dcsg --dataset amazon --device cuda --cuda 0

# Plain LightGCN baseline using its own configuration
python encoder/train_encoder.py --model lightgcn --dataset amazon --device cuda --cuda 0
```

Important configuration fields are collected in the selected YAML file:

| Field | Meaning |
| --- | --- |
| `train.epoch`, `train.batch_size`, `train.patience` | Maximum epochs, batch size, and early-stopping patience |
| `test.k` | Evaluation cutoffs; supplied DCSG configurations use 5, 10, and 20 |
| `model.embedding_size` | Collaborative representation dimension |
| `model.{dataset}.layer_num` | Number of graph propagation layers for that dataset |
| `sem_graph_topk` | Semantic user-item neighbors retained per user |
| `keep_rate` | Edge-retention ratio for the structural view |
| `contrastive_weight`, `kd_weight` | Weights for consistency and semantic distillation terms |
| `gate_supervision_weight` | Weight of the optional gate supervision loss |

Training validates every `test_step`, early-stops on the final Recall cutoff, then reports final test metrics using the best validation model.

## Evaluation and Outputs

The trainer performs all-rank evaluation after training, masks observed training interactions, and reports the YAML metrics, normally Recall@{5,10,20} and NDCG@{5,10,20}.

For a completed run, artifacts are written to:

```text
encoder/checkpoint/{model}/{model}-{dataset}-{seed}.pth  # Best model parameters
encoder/log/{model}/{dataset}_{timestamp}.log            # Configuration and metrics log
candidate.txt                                             # Top-100 item IDs per test user
```

`candidate.txt` is overwritten by the next run; rename it before another experiment when it is needed for analysis.

## Optional LLM-Guided Gate Labels

DCSG can use GPT-4o-mini-generated labels to supervise its fusion gate. The prompt returns one number in `[0, 1]`: `0` denotes complete reliance on semantic information and `1` denotes complete reliance on collaborative information. Default configurations set `preprocess_gate_labels: false`, so normal training uses initialized neutral labels and makes no API calls.

To prepare labels, configure the LLM locally, provide valid profile JSON files, and set `preprocess_gate_labels: true`. Labels are cached as `llm_gate_labels_{dataset}.pt` in the working directory. Since preprocessing queries individual nodes, run it offline.

Before enabling this option, verify the profile path used by `LLMGuideManager`: the current implementation resolves profiles through a relative `../data/{dataset}/` path, while the standard training command is run from the repository root. Adjust that path for the local working directory before using LLM label preprocessing.

Do not commit API keys, cached private profiles, or generated labels that are subject to a provider's data-use restrictions.

The scripts in `generation/` demonstrate profile prompting and embedding. Adapt their input/output loops when regenerating a full profile corpus.

## Reproducibility Notes

- Use `--seed` to override the YAML seed. Set the same seed, model configuration, data artifacts, and CUDA environment when comparing methods.
- The supplied configuration uses up to 3000 epochs with validation every three epochs and patience five. Early stopping usually ends training earlier.
- Batch size and evaluation batch size are hardware-dependent. Lower them in the YAML file when GPU memory is limited.
- Profile embeddings, interaction-ID mappings, and dataset splits are part of the experimental input. Re-generating any of them changes the experiment.

## Citation

If you use this repository, please cite the accompanying paper. A BibTeX entry will be added once the paper metadata is finalized.

## License and Data Use

This repository does not currently include a standalone license file. Please follow the original licenses and terms of use of the Amazon, Yelp, and Steam data sources, the LLM or embedding providers, and all baseline implementations.
