# TriBind
A Tri-Branch Graph Neural Network Integrating Protein Surface Features and Interaction Geometry for Binding Affinity Prediction

## :ledger: Index

- [About](#Note)
- [Dataset](#Dataset)
- [Requirements](#Requirements) 
- [Repository Structure](#Repository-Structure)
- [Usage](#Usage)
  - [Data preprocessing](#1.-Data-preprocessing)
  - [Model training](#2.-Model-training)
  - [Model evaluation](#3.-Model-evaluation)
  - [External prediction](#4.-External-prediction) 
- [Reproducibility](#Reproducibility)
- [Citation](#Citation)
- [Acknowledgements](#Acknowledgements)


# Note: 

This repository contains the official implementation of TriBind, a tri-branch graph neural network for protein–ligand binding affinity prediction that integrates:

Protein surface representations extracted from residue-level geometric and physicochemical descriptors of MaSIF (https://github.com/LPDI-EPFL/masif-neosurf).
Protein–ligand interaction graphs encoding explicit intermolecular interactions (Interaction_parser_NEW.py).
Protein coarse-grained molecular graphs include residue-level chemical information (Protein_parser.py).
Ligand molecular graphs capturing atom-level chemical information (mol_parser.py).

The framework is designed to learn complementary information from molecular topology, interaction patterns, and protein surface properties, resulting in accurate and interpretable binding affinity prediction.

Several baseline models used in this study are adapted from their original implementations. We sincerely thank the original authors for making their code publicly available.

PotentialNet: https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/model/model_zoo/potentialnet.py 

GIGN: https://github.com/guaguabujianle/GIGN/tree/main 

Pafnucy: https://github.com/realfolkcode/Pafnucy 

DeepDTA: https://github.com/hkmztrk/DeepDTA 

MGraphDTA: https://github.com/guaguabujianle/MGraphDTA

All scoring functions (SFs) calculated in Discovery Studio v4.1 software.

# Dataset:

All datasets used in this work are publicly available.

PDBbind
PDBbind v2020: http://www.pdbbind.org.cn/download.php

CASF Benchmarks
CASF-2013
CASF-2016
CASF-2019

http://www.pdbbind.org.cn/casf.php

External Validation Dataset:

The external α-glucosidase inhibitor dataset used in this work was collected from the published literature and prepared following the protocol described in our paper (https://www.sciencedirect.com/science/article/pii/S2405580825000822).

# Requirements

The implementation is developed with Python and PyTorch.

Example environment:

python==3.10

torch==2.x
torch_geometric==2.x
numpy
scipy
pandas
scikit-learn
matplotlib
networkx
rdkit
biopython
biopandas
oddt
tqdm
pyyaml
joblib

###  :file_folder: Repository Structure

```
.
TriBind
├── data/
│   ├── train/
│   ├── valid/
│   ├── test2013/
│   ├── test2016/
│   ├── test2019/
│   └── external_test/
│       ├── index.csv
│       └── results.csv
│
├── preprocessing/
│
├── TB-PLI+Surf/
│   ├── results/
│   ├── model/
│   └── test/
├── DB-PL+Surf/
│   └── ...
├── TB-PLI/
│   └── ...
├── SB-PLI+Surf/
│   └── ...
│
├── scripts/
├── results/
├── requirements.txt
└── README.md
```


# Usage
1. Data preprocessing

Prepare the raw protein–ligand complexes following the directory structure described above.

Generate:

Protein surface descriptors with MaSIF.
Protein–ligand interaction graphs
Ligand molecular graphs
python preprocessing.py

Then construct the PyTorch Geometric datasets

python build_dataset.py


2. Model training

Train each model

python train.py

Training parameters such as learning rate, batch size, hidden dimension, and random seed can be modified in the related file.


3. Model evaluation

Evaluate a trained checkpoint on CASF benchmark datasets

python test.py

The evaluation script reports

Pearson correlation (R)
Spearman correlation (ρ)
RMSE
MAE
SD


4. External prediction

To predict binding affinities for unseen protein–ligand complexes

Organize data as

data/
└── external_test/
      └── PDB_ID/
            ├── protein.pdb
            ├── ligand.mol2
            └── ligand.sdf

Run

1) python preprocessing.py

2) python build_dataset.py

3) python predict.py

Predicted binding affinities will be saved in

results/predictions.csv


# Reproducibility

To reproduce the results reported in the paper:

Download the PDBbind datasets.
Generate protein surface descriptors and protein-ligand complex + interaction graphs.
Train the model using the provided configuration.
Evaluate on the CASF-2013, 2016, and 2019 benchmark sets.
Test on the external α-glucosidase inhibitor dataset.

Random seeds are fixed to ensure reproducibility.

# Citation

If you find this work useful in your research, please cite

@article{
}

# Acknowledgements
