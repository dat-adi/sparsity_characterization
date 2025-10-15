import torch

import triton
import triton.language as tl

def init_to_zero(*names):
    def init_func(nargs):
        for name in names:
            nargs[name].zero_()
    return init_func

DEVICE = torch.device("cuda:0")

configs=[
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=2, pre_hook=init_to_zero("Y")), 
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}, num_warps=2, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 16}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
]

@triton.autotune(
    configs=configs,
    key=["CACHE_KEY_M", "CACHE_KEY_N", "BATCHSIZE", "SPARSITY_BIN"],
)
@triton.jit
def splitk_sparse_gemv_kernel(
    Y, # Pointers to matrices
    A, X, threshold,
    # Matrix dimensions
    N, M,
    CACHE_KEY_N, CACHE_KEY_M,
    # Meta-parameters
    BATCHSIZE: tl.constexpr, SPARSITY_BIN: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
):
    start_n = tl.program_id(0)
    start_m = tl.program_id(1)
    # now compute the block that each program will go through
    # rn (resp. rm) denotes a range of indices for rows (resp. col) of A
    
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    A_ptr = A + (rm[:, None] * N + rn[None, :])
    X_ptr = X + rm
    Y_ptr = Y + rn

    # eviction policy go brrr
    if BATCHSIZE == 1:
        x0 = tl.load(X_ptr, mask=rm < M, other=0.0, eviction_policy='evict_last') # reuse x across threadblocks
        idx = tl.abs(x0) > threshold
        # selectively load weight rows
        a = tl.load(A_ptr, mask=idx[:, None], other=0.0, eviction_policy='evict_first') # only load weights once per threadblock
        acc0 = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)

    # rematerialize rm and rn to save registers
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    tl.atomic_add(Y_ptr, acc0, mask=rn < N)


# NOTE: assumes that weight is column major
def splitk_sparse_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    threshold: float,
    sparsity_bin: int
) -> torch.Tensor:
    """
    Compute y = sparse(X) @ weight.
    :param x: input tensor [1, 1, Z]
    :param weight: weight matrix [N, Z]
    :param threshold: threshold for the absolute value of x
    :param sparsity_bin: sparsity level to get tuned kernel
    :return: result tensor y
    """
    N, Z = weight.shape
    beam_width, seq_len, _ = x.shape
    assert x.shape[2] == Z
    x = x.contiguous()
    
    assert weight.stride(1) > 1, "weight should be column major"

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(N, META["BLOCK_N"]),
        triton.cdiv(Z, META["BLOCK_M"]),
    )  # noqa

    output = torch.empty(
        beam_width,
        seq_len,
        N,
        device=x.device,
        dtype=torch.float16,
    )


    kernel = splitk_sparse_gemv_kernel
    kernel[grid](
        output,  # data ptrs
        weight,
        x,
        threshold,
        N,  # shapes
        Z,
        N // 16,  # key for triton cache (limit number of compilations)
        Z // 16,
        beam_width,  # BATCHSIZE
        sparsity_bin, # SPARSITY_BIN
        # can't use kwargs because auto-tuner requires args
    )

    if x.dtype is not output.dtype:
        print(f"Warning: incuring dtype conversion overhead since input dtype is not torch.float16. Detected dtype: {x.dtype}. ")
        return output.to(dtype=x.dtype)

    return output

torch.manual_seed(0)
threshold = 0.02037687972187996

w1 = torch.load("./weights.pt")
w1 = w1.to(device=DEVICE, dtype=torch.float16)
w1 = w1.data.T.contiguous().T # column major

x = torch.load("./x.pt")
x = x.to(device=DEVICE, dtype=torch.float16)

print(x.shape, " | ", w1.shape)
result = splitk_sparse_gemv(x, w1, threshold, 0)
print(result)

x = x[0]

mask = (x.abs() > threshold).float()
masked = (x * mask).to(dtype=torch.float16)
output = torch.empty(1, 6, 11008, device=DEVICE, dtype=torch.float16)
# TODO: Check how it operates for prefill.
for i in range(mask.shape[0]):
    masked_A = (w1 * mask[i, :])
    output[i] = masked_A.sum(axis=0) * x[i, :]

print(output)
