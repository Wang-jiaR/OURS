# Exploiting Cross-Modal Duality in Backdoor Attacks on Medical Foundation Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/PyTorch-v1.12%2B-red)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Under_Review-blue)](https://anonymous.4open.science/)

This repository contains the official implementation of the paper: **"Exploiting Cross-Modal Duality in Backdoor Attacks on Medical Foundation Models"**.

## 📌 Introduction

Medical foundation models (e.g., BioMedCLIP, PLIP) leverage cross-modal associations for superior transfer learning. However, we posit that this **cross-modal architecture constitutes both a performance strength and a critical security vulnerability**.

We propose a novel **Cross-Modal Backdoor Framework** inspired by human multisensory integration. Unlike conventional attacks requiring large-scale retraining, our approach constructs implicit triggers in the feature space via semantic inversion.

### ✨ Key Features
*   **Dual Optimization Strategy:** Combines imperceptible visual perturbations (for stealth) with parameter-efficient medical semantic prompts (to recalibrate modality alignment).
*   **Dynamic Trigger Generation:** Utilizes Optimization-based Text Inversion (OTI) to generate triggers that are semantically consistent with medical logic.
*   **High Efficiency:** Achieves near-perfect attack success (ASR > 99%) with only **5% poisoning rate** and minimal parameter modifications.
*   **Robustness:** Validated across 3 foundation models (BioMedCLIP, PLIP, QuiltNet) and 6 datasets (Medical & Natural).

## 🛠️ Environment Setup

To set up the environment, please run the following commands:

```bash
conda create -n cross_modal_attack python=3.9
conda activate cross_modal_attack
pip install -r requirements.txt
Key Dependencies:
torch >= 1.12.0
open_clip_torch
transformers
scikit-learn
pandas

## 📂 Data Preparation

We perform experiments on three medical datasets and three general vision datasets. Please organize your data as follows:

```text
./data/
├── Kather/           # Histology dataset (CRC-DX)
├── PanNuke/          # Nuclei instance segmentation/classification
├── DigestPath/       # Digestive system pathology
├── MNIST/
├── CIFAR10/
└── COCO2017/

---

### 2. Usage 部分（代码运行命令）

```markdown
## 🚀 Usage

### 1. Pre-trained Foundation Models
Ensure you have the weights for the target foundation models. Our code supports:
*   **BioMedCLIP**: [HuggingFace](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
*   **PLIP**: [HuggingFace](https://huggingface.co/vinid/plip)
*   **QuiltNet**: [HuggingFace](https://huggingface.co/nurs/quiltnet)

### 2. Stage 1: Dynamic Trigger Generation
Generate the semantic implicit triggers using gradient-guided inversion (OTI).

```bash
python generate_trigger.py \
  --model PLIP \
  --dataset Kather \
  --steps 150 \
  --prompt_len 16

python train_backdoor.py \
  --model PLIP \
  --dataset Kather \
  --poison_rate 0.05 \
  --batch_size 16 \
  --lr 5e-5 \
  --epsilon 8 \
  --alpha 0.02
