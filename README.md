# DCSG

## Dual Channel Learning Enhanced with Semantic Graph for Recommendation

This repository contains the implementation of **DCSG** (*Dual Channel Learning Enhanced with Semantic Graph for Recommendation*), a graph-based recommender system that combines collaborative interaction signals with LLM-derived semantic representations of users and items.

The project targets top-k recommendation on the Amazon, Yelp, and Steam datasets. It includes the DCSG implementation built on several collaborative-filtering backbones, data artifacts used by the main experiments, profile-generation examples, and ablation utilities.

> The paper manuscript corresponding to this code is available in the repository root as `_KBS__Dual_Channel_Learning_Enhanced_with_Semantic_Graph_for_Recommendation (3).pdf`.

## Overview

Collaborative filtering is effective when interaction data are dense, but it can struggle to express semantic relations that are absent from the observed user-item graph. DCSG introduces a semantic channel alongside the collaborative channel:

1. **Collaborative channel.** A graph recommender propagates trainable user and item embeddings over the normalized user-item interaction graph.
2. **Semantic channel.** LLM-generated user and item profiles are encoded as dense embeddings. DCSG builds a weighted user-item semantic graph by retaining each user's top-`k` most similar items, then applies graph attention to the projected profile embeddings.
3. **Adaptive fusion.** A feature-wise sigmoid gate learns to combine collaborative and semantic representations for every user and item.
4. **Joint optimization.** The default DCSG variants optimize BPR ranking loss together with embedding regularization, semantic knowledge distillation, and contrastive consistency. The gate can additionally be supervised by cached LLM-generated labels.

The default main configuration is `lightgcn_dcsg`, which uses LightGCN as the collaborative backbone. DCSG counterparts are also provided for SGL, SimGCL, GCCF, GraphAU, NCL, SGCF, DirectAU, AdaGCL, and AutoCF where corresponding files and configurations are present.

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

Files outside these paths are not part of the DCSG experimental pipeline.

## Environment

The code is written in Python and the DCSG implementation currently relies on CUDA tensors internally. Use a CUDA-enabled PyTorch environment for the main experiments.

Recommended setup:

```bash
conda create -n dcsg python=3.10 -y
conda activate dcsg

# Install the PyTorch build matching the local CUDA version first.
pip install torch

pip install numpy scipy scikit-learn pyyaml tqdm requests openai matplotlib
pip install torch-geometric torch-sparse
```

`torch-geometric` and `torch-sparse` must match the installed PyTorch and CUDA versions. Please use the installation matrix published by PyTorch Geometric when the simple `pip` installation does not provide a compatible wheel.

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

`trn_mat.pkl`, `val_mat.pkl`, and `tst_mat.pkl` must deserialize to matrices with the same shape: `[num_users, num_items]`. The training matrix defines the ID space. Profile embedding rows must align exactly with these user and item indices. The current semantic projection layer expects **1536-dimensional** profile embeddings.

The repository already contains the processed artifacts for the three supported datasets. If preparing a new dataset, construct the interaction splits and aligned profile/embedding files in the same format, then add its dataset-specific block to the selected YAML configuration. The current data handler explicitly restricts DCSG loading to Amazon, Yelp, and Steam.

### Semantic Graph Construction

At load time, DCSG L2-normalizes user and item profile embeddings, computes user-item cosine similarities, and retains the `sem_graph_topk` highest-scoring items for each user. The retained similarities become weighted edges of the semantic graph. `sem_graph_topk` is configured per dataset in `encoder/config/modelconf/*_dcsg.yml` and is `10` in the supplied main configurations.

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

The `--model` argument selects `encoder/config/modelconf/{model}.yml`; the model is then dynamically loaded from `encoder/models/general_cf/{model}.py`. Useful examples are:

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
| `train.epoch`, `train.batch_size`, `train.patience` | Maximum epochs, mini-batch size, and early-stopping patience |
| `test.k` | Evaluation cutoffs; supplied DCSG configurations use 5, 10, and 20 |
| `model.embedding_size` | Collaborative representation dimension |
| `model.{dataset}.layer_num` | Number of graph propagation layers for that dataset |
| `sem_graph_topk` | Semantic user-item neighbors retained per user |
| `keep_rate` | Edge-retention ratio for the structural view |
| `contrastive_weight`, `kd_weight` | Weights for consistency and semantic distillation terms |
| `gate_supervision_weight` | Weight of the optional gate supervision loss |

Training prints validation metrics at each configured `test_step`, performs early stopping using the final Recall cutoff, reloads the best validation checkpoint in memory, and reports final test metrics.

## Evaluation and Outputs

The trainer performs all-rank evaluation after training. It masks training interactions before ranking and reports the metrics listed in the YAML configuration, normally Recall@{5,10,20} and NDCG@{5,10,20}.

For a completed run, artifacts are written to:

```text
encoder/checkpoint/{model}/{model}-{dataset}-{seed}.pth  # Best model parameters
encoder/log/{model}/{dataset}_{timestamp}.log            # Configuration and metrics log
candidate.txt                                             # Top-100 item IDs per test user
```

The `candidate.txt` output is overwritten by the next run. Preserve or rename it before starting another experiment if it is needed for external analysis.

## Optional LLM-Guided Gate Labels

DCSG can use LLM-generated labels to supervise its fusion gate. The default DCSG configurations set `preprocess_gate_labels: false`, so normal training uses the initialized neutral gate labels and does not make API calls.

To prepare LLM labels, provide valid user and item profile JSON files, configure the LLM settings locally in the selected YAML file, and set `preprocess_gate_labels: true`. The model will cache labels as `llm_gate_labels_{dataset}.pt` in the process working directory. This preprocessing can be expensive because it queries a model for individual nodes; it is best treated as an offline step.

Before enabling this option, verify the profile path used by `LLMGuideManager`: the current implementation resolves profiles through a relative `../data/{dataset}/` path, while the standard training command is run from the repository root. Adjust that path for the local working directory before using LLM label preprocessing.

Do not commit API keys, cached private profiles, or generated labels that are subject to a provider's data-use restrictions.

The scripts under `generation/` show the prompting and embedding workflow used to derive semantic inputs. They are demonstration scripts rather than a dataset-scale end-to-end preprocessing command: adapt their input/output loops when regenerating complete profile corpora.

## Ablations

### Backbone and component comparisons

Use the relevant configuration and model name to compare DCSG with its backbone:

```bash
python encoder/train_encoder.py --model lightgcn --dataset yelp --device cuda --cuda 0
python encoder/train_encoder.py --model lightgcn_dcsg --dataset yelp --device cuda --cuda 0
```

Additional comparison configurations include `*_wogat_*` and `*_wogate_*` variants. Their exact behavior is defined by the corresponding model files and YAML settings; retain the same dataset split and seed when reporting an ablation.

### Topology runner

`encoder/scripts/run_topology_ablation.py` launches repeated training runs and collects Recall@10 and NDCG@10 into `encoder/experiments/topology_ablation/`:

```bash
python encoder/scripts/run_topology_ablation.py \
  --datasets yelp \
  --topologies user_item \
  --seeds 2025 \
  --device cuda \
  --cuda 0
```

The current data loader constructs the user-item semantic graph. Although the runner accepts `item_item` and `user_user`, the supplied graph-construction implementation does not yet branch on `semantic_graph_topology`; implement those graph builders before treating those two options as topology ablation results.

## Reproducibility Notes

- Use `--seed` to override the YAML seed. Set the same seed, model configuration, data artifacts, and CUDA environment when comparing methods.
- The supplied configuration uses up to 3000 epochs with validation every three epochs and patience five. Early stopping usually ends training earlier.
- Batch size and evaluation batch size are hardware-dependent. Lower them in the YAML file when GPU memory is limited.
- Profile embeddings, interaction-ID mappings, and dataset splits are part of the experimental input. Re-generating any of them changes the experiment.
- CPU execution is not currently supported by the DCSG path because semantic graph and profile tensors use CUDA directly.

## Citation

If you use this repository, please cite the accompanying paper. A BibTeX entry will be added once the paper metadata is finalized.

## License and Data Use

This repository does not currently include a standalone license file. Please follow the original licenses and terms of use of the Amazon, Yelp, and Steam data sources, the LLM or embedding providers, and all baseline implementations.
