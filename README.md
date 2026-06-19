# TriBind
A Tri-Branch Graph Neural Network Integrating Protein Surface Features and Interaction Geometry for Binding Affinity Prediction

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
