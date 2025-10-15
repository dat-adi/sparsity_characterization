import torch
from pathlib import Path

sparsities = { # 0.9
        'q': 0.018186461180448532,
        'k': 0.018186461180448532,
        'v': 0.018186461180448532,
        'o': 0.0034913804847747087,
        'up': 0.06005464494228363,
        'gate': 0.06005464494228363,
        'down': 0.003233462106436491
}

# Step 1: Load the .pt file
for f in Path.cwd().joinpath("original_0.9_workloads").iterdir():
    try:
        if f.suffix != ".pt":
            continue

        file_path = str(f)
        matrix = torch.load(file_path)

        if f.name.startswith("A"):
            mask = matrix > sparsities["gate"]
        elif f.name.startswith("B"):
            mask = matrix > sparsities["up"]
        elif f.name.startswith("C"):
            mask = matrix > sparsities["down"]
        else:
            continue

        matrix = matrix.masked_fill(mask, 0)

        total_positions = matrix.numel()
        non_zeros = torch.count_nonzero(matrix)
        num_zeros = total_positions - non_zeros
        print(f"Number of zeros: {num_zeros} / {total_positions} # {f.name}")

        torch.save(matrix, Path.cwd() / "sparsified_versions" / f.name)
    except Exception as err:
        print(f.name, " has failed due to ", err)

# Step 2: Convert to numpy array
# matrix = matrix.cpu().numpy()

# Step 3: Create the sparsity plot
# plt.figure(figsize=(10, 10))
# plt.spy(matrix, markersize=1)  # Use spy for sparsity plotting
# plt.title("Sparsity Matrix")
# plt.xlabel("Columns")
# plt.ylabel("Rows")
# plt.savefig('C_matrix_1d36f97e.png')
