import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

for f in Path.cwd().iterdir():
    try:
        if f.suffix != ".pt": continue
        file_name = str(f.name)
        matrix = torch.load(file_name, weights_only=True)
        total_positions = matrix.numel()
        non_zeros = torch.count_nonzero(matrix)
        num_zeros = total_positions - non_zeros
        print(f"Number of zeros: {num_zeros} / {total_positions} # {file_name}")

        matrix = matrix.cpu().numpy()

        plt.figure(figsize=(100, 100), dpi=80)
        plt.spy(matrix)  # Use spy for sparsity plotting
        plt.title("Sparsity Matrix")
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.savefig(file_name[:-3] + ".png")

    except Exception as err:
        print(f.name, " has failed due to ", err)
