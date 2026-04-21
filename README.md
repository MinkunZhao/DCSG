# Dual Channel Learning Enhanced with Semantic Graph for Recommendation (DCSG)

This repository contains the implementation of the Dual Channel Learning Enhanced with Semantic Graph (DCSG) model for recommendation systems.

## Overview

The DCSG model integrates dual-channel learning with semantic graph enhancement to improve recommendation performance. The model combines structural information from collaborative filtering with semantic information from user and item profiles, enhanced by Large Language Model (LLM) supervision signals.


## Usage

### Training a Model

1. Configure your model in the appropriate YAML file
2. Run the training script:
   ```bash
   python encoder/train_encoder.py --config encoder/config/modelconf/lightgcn_dcsg.yml
   ```