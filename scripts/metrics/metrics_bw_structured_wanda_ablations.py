import sys
import torch
from pathlib import Path
from rich import print

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.similarity_metrics import compute_metrics_by_feature

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
