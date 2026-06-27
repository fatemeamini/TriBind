import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
import copy

# ── Import your model ──────────────────────────────────────────────
from ProteinLigandInteractionModel_scale_modified_2 import TriBranchDTI


class GradAAM:
    def __init__(self, model, module):
        self.model = model
        self.module = module
        self._gradients = None
        self._activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            # GATConv returns a tensor directly
            if isinstance(output, tuple):
                self._activations = output[0].detach()
            else:
                self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                self._gradients = grad_output[0].detach()
            else:
                self._gradients = grad_output.detach()

        self.module.register_forward_hook(forward_hook)
        self.module.register_full_backward_hook(backward_hook)

    def __call__(self, prot_data, lig_data, int_data):
        self.model.zero_grad()
        pred = self.model(prot_data, lig_data, int_data)
        pred.backward()

        grads = self._gradients      # [N, C]
        acts  = self._activations    # [N, C]

        if grads is None or acts is None:
            raise RuntimeError("Hooks did not fire — check module name")

        print(f"  grads min/max: {grads.min():.4f} / {grads.max():.4f}")
        print(f"  acts  min/max: {acts.min():.4f} / {acts.max():.4f}")

        weights = grads.mean(dim=0, keepdim=True)
        cam = (weights * acts).sum(dim=1)
        cam = cam.abs() 

        cam_np = cam.cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np, float(pred.item())