from pathlib import Path

def get_unstructured_matrices() -> tuple[list[Path], list[Path]]:
    # Define paths
    wanda_dir = Path("/home/datadi/burns/aws/workloads/data/wanda_unstructured/layer-1")
    sparsegpt_dir = Path("/home/datadi/burns/aws/workloads/data/sparsegpt_unstructured/layer-1")

    # Matrix files to analyze
    matrix_files = [
        "layer1-mlp.down_proj.pt",
        "layer1-mlp.up_proj.pt",
        "layer1-mlp.gate_proj.pt",
        "layer1-self_attn.q_proj.pt",
        "layer1-self_attn.k_proj.pt",
        "layer1-self_attn.v_proj.pt",
        "layer1-self_attn.o_proj.pt",
    ]

    wanda_matrices = [wanda_dir / matrix_file for matrix_file in matrix_files]
    sparsegpt_matrices = [sparsegpt_dir / sd for sd in matrix_files]

    return wanda_matrices, sparsegpt_matrices

def set_seed(random_seed: int) -> None:
    import random
    import numpy as np
    import torch
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
