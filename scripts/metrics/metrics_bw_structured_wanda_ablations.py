import torch
from pathlib import Path
from rich import print
import numpy as np

u24 = {
"wanda": [ # unstructured
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-mlp.down_proj-0-0.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-mlp.gate_proj-0-0.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-mlp.up_proj-0-0.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-self_attn.k_proj-0-0.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-self_attn.o_proj-0-0.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-self_attn.q_proj-0-0.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-self_attn.v_proj-0-0.pt",
],
"sparsegpt": [
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-mlp.down_proj-2-4.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-mlp.gate_proj-2-4.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-mlp.up_proj-2-4.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-self_attn.k_proj-2-4.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-self_attn.o_proj-2-4.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-self_attn.q_proj-2-4.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-self_attn.v_proj-2-4.pt"
]
}

u48 = {
"wanda": [ # unstructured
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-mlp.down_proj-0-0.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-mlp.gate_proj-0-0.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-mlp.up_proj-0-0.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-self_attn.k_proj-0-0.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-self_attn.o_proj-0-0.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-self_attn.q_proj-0-0.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-self_attn.v_proj-0-0.pt",
],
"sparsegpt": [
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer0-mlp.down_proj-4-8-0.5.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer0-mlp.gate_proj-4-8-0.5.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer0-mlp.up_proj-4-8-0.5.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer0-self_attn.k_proj-4-8-0.5.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer0-self_attn.o_proj-4-8-0.5.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer0-self_attn.q_proj-4-8-0.5.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer0-self_attn.v_proj-4-8-0.5.pt",
]
}

tf_48 = {
"wanda": [ # unstructured
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-mlp.down_proj-2-4.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-mlp.gate_proj-2-4.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-mlp.up_proj-2-4.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-self_attn.k_proj-2-4.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-self_attn.o_proj-2-4.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-self_attn.q_proj-2-4.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer1-self_attn.v_proj-2-4.pt"
],
"sparsegpt": [
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer0-mlp.down_proj-4-8-0.5.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer0-mlp.gate_proj-4-8-0.5.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer0-mlp.up_proj-4-8-0.5.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer0-self_attn.k_proj-4-8-0.5.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer0-self_attn.o_proj-4-8-0.5.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer0-self_attn.q_proj-4-8-0.5.pt",
"/home/datadi/burns/aws/workloads/data/ablations/structure_layer_1_wanda/layer0-self_attn.v_proj-4-8-0.5.pt",
]
}

def jaccard_similarity(a, b):
    a_nz = (a != 0).flatten()
    b_nz = (b != 0).flatten()
    intersection = torch.sum(a_nz & b_nz).item()
    union = torch.sum(a_nz | b_nz).item()
    return intersection / union if union > 0 else 0

def cosine_similarity(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    dot = torch.dot(a_flat, b_flat)
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)
    return (dot / (norm_a * norm_b)).item() if norm_a > 0 and norm_b > 0 else 0

def hamming_distance(a, b):
    a_nz = (a != 0).flatten()
    b_nz = (b != 0).flatten()
    return torch.sum(a_nz != b_nz).item() / len(a_nz)

def compute_metrics_by_feature(mat1, mat2, matrix_name):
    is_down_proj = "down_proj" in matrix_name
    axis = 0 if is_down_proj else 1  # row-wise for down_proj, column-wise for others
    
    jaccard_scores = []
    cosine_scores = []
    hamming_scores = []
    
    if axis == 0:  # row-wise
        for i in range(mat1.shape[0]):
            jaccard_scores.append(jaccard_similarity(mat1[i], mat2[i]))
            cosine_scores.append(cosine_similarity(mat1[i], mat2[i]))
            hamming_scores.append(hamming_distance(mat1[i], mat2[i]))
    else:  # column-wise
        for i in range(mat1.shape[1]):
            jaccard_scores.append(jaccard_similarity(mat1[:, i], mat2[:, i]))
            cosine_scores.append(cosine_similarity(mat1[:, i], mat2[:, i]))
            hamming_scores.append(hamming_distance(mat1[:, i], mat2[:, i]))
    
    return {
        'jaccard': {'mean': np.mean(jaccard_scores), 'std': np.std(jaccard_scores)},
        'cosine': {'mean': np.mean(cosine_scores), 'std': np.std(cosine_scores)},
        'hamming': {'mean': np.mean(hamming_scores), 'std': np.std(hamming_scores)}
    }

def comparison_between_wanda_and_sparsegpt_unstructured(file_paths):
    matrix_count = len(file_paths["wanda"])
    results = {}
    
    for i in range(matrix_count):
        mat1 = torch.load(file_paths["wanda"][i])
        mat2 = torch.load(file_paths["sparsegpt"][i])
        
        matrix_name = Path(file_paths["wanda"][i]).name.split("-")[1:3]
        matrix_name = "-".join(matrix_name).split("-0-0-0.5.pt")[0]
        
        metrics = compute_metrics_by_feature(mat1, mat2, matrix_name)
        results[matrix_name] = metrics
        
        print(f"\n{matrix_name}:")
        print(f"  Jaccard: {metrics['jaccard']['mean']:.4f} ± {metrics['jaccard']['std']:.4f}")
        print(f"  Cosine:  {metrics['cosine']['mean']:.4f} ± {metrics['cosine']['std']:.4f}")
        print(f"  Hamming: {metrics['hamming']['mean']:.4f} ± {metrics['hamming']['std']:.4f}")
    
    return results

print("Unstructured and 2:4")
results = comparison_between_wanda_and_sparsegpt_unstructured(u24)

print("Unstructured and 4:8")
results = comparison_between_wanda_and_sparsegpt_unstructured(u48)

print("2:4 and 4:8")
results = comparison_between_wanda_and_sparsegpt_unstructured(tf_48)
