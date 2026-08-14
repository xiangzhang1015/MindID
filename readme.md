<div align="center">

# MindID: Person Identification from Brain Waves through Attention-based Recurrent Neural Network

[![UbiComp 2018](https://img.shields.io/badge/UbiComp-2018-blue.svg)](https://dl.acm.org/doi/10.1145/3264959)
[![arXiv](https://img.shields.io/badge/arXiv-1711.06149-b31b1b.svg)](https://arxiv.org/abs/1711.06149)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](LICENSE)

**Xiang Zhang, Lina Yao, Salil S. Kanhere, Yunhao Liu, Tao Gu, Kaixuan Chen**

*Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT), Vol. 2, No. 3, Article 149, 2018*

</div>

---

## Overview

Traditional biometric identification methods are vulnerable to spoofing and replay attacks. MindID addresses this by using EEG brainwave signals, which are inherently attack-resilient and cannot be forged without the subject's presence. The method isolates the Delta frequency band (0.5-4 Hz) as the most subject-discriminative component, then feeds the decomposed signals through an attention-based encoder-decoder RNN that assigns channel-wise importance weights before classifying identity. The system achieves 0.982 accuracy on the EID-M dataset across 8 subjects.

## Method

```
Raw EEG Signal
      |
      v
[Delta Band Decomposition]   Butterworth bandpass filter (0.5-4 Hz)
      |
      v
[Input Projection Layer]     Sigmoid-activated fully connected layers
      |
      v
[LSTM Encoder]               BasicLSTMCell; hidden state captures temporal dynamics
      |
      v
[Attention Module]           Element-wise multiplication of output and cell state
      |
      v
[Classification Head]        Linear projection -> softmax over 8 subject classes
      |
      v
[XGBoost Re-classifier]      Deep features extracted from LSTM fed into XGBoost
      |
      v
Predicted Subject ID
```

- **Delta decomposition:** A 3rd-order Butterworth bandpass filter isolates the Delta pattern, which the paper identifies as carrying the most discriminative information for person identification.
- **Attention-based RNN:** The final cell state of the LSTM acts as a learned attention weight, modulating the output representation before classification.
- **Two-stage classifier:** The LSTM serves as a feature extractor; XGBoost operates on the extracted representations as a second-stage classifier.
- **Preprocessing:** DC offset subtraction (4200 units) followed by z-score normalization per channel.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

| Package | Role |
|---|---|
| `tensorflow==1.14.0` | LSTM model training and inference |
| `numpy` | Array operations and data manipulation |
| `scipy` | Butterworth bandpass filter (`butter`, `lfilter`) |
| `scikit-learn` | z-score normalization (`preprocessing.scale`) |
| `xgboost` | Second-stage classifier on deep features |

> **Note:** The code uses TensorFlow 1.x APIs (`tf.contrib`, `tf.Session`, `tf.placeholder`) and Python 2 print syntax. A Python 2.7 environment with TensorFlow 1.x is required to run it as-is.

## Data

See [data/README.md](data/README.md) for full dataset descriptions and download instructions.

Three datasets are supported:

| Dataset | Type | Subjects | Channels | Samples/Subject |
|---|---|---|---|---|
| EID-M | Local (provided) | 8 | 14 | 21,000 (3 trials) |
| EID-S | Local (provided) | 8 | 14 | 7,000 (1 trial) |
| EEG-S | Public (PhysioNet eegmmidb) | 8 | 64 | 13,500 (1 trial) |

EID-S is used by default. Place the `.mat` files in the project root before running.

## Usage

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Place data files in the project root**

```
MindID/
  EID-M.mat
  EID-S.mat
```

**3. Run the model**

```bash
python main.py
```

**4. Switch datasets (optional)**

Edit `src/model.py` lines 46-49 to select EID-M or EEG-S. See [data/README.md](data/README.md) for details.

## Repository Structure

```
MindID/
  main.py                  # Entry point
  requirements.txt         # Python dependencies
  CITATION.cff             # Machine-readable citation
  LICENSE                  # CC BY-NC-SA 4.0
  .gitignore
  src/
    __init__.py
    model.py               # Delta decomposition, attention RNN, XGBoost pipeline
  data/
    README.md              # Dataset descriptions and download instructions
  Flowchart of the proposed approach_MindID.PNG
```

## Citation

```bibtex
@article{zhang2018mindid,
  title     = {{MindID}: Person Identification from Brain Waves through Attention-based Recurrent Neural Network},
  author    = {Zhang, Xiang and Yao, Lina and Kanhere, Salil S. and Liu, Yunhao and Gu, Tao and Chen, Kaixuan},
  journal   = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume    = {2},
  number    = {3},
  pages     = {1--23},
  year      = {2018},
  publisher = {ACM},
  doi       = {10.1145/3264959}
}
```

## Contact

For questions about the code or algorithm, contact [xiang.alan.zhang@gmail.com](mailto:xiang.alan.zhang@gmail.com).

## License

This work is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](LICENSE).
