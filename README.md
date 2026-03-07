# MatGL Surface-Aware (CHGNet)

This repository is forked from [materialsvirtuallab/matgl](https://github.com/materialsvirtuallab/matgl).

For the original MatGL framework, APIs, and model zoo, please refer to the upstream repository and docs:
- Upstream repo: https://github.com/materialsvirtuallab/matgl
- Documentation: https://matgl.ai

## What We Changed

This fork focuses on a **surface-aware CHGNet** design for slab/interface/catalysis scenarios, while preserving differentiability and energy-force consistency.

### 1. Surface Descriptors in Feature Construction

We introduce two local descriptors computed from neighbor geometry:

- `surface_degree u_i`: continuous exposure/undercoordination score (`0 ~ bulk-like`, `1 ~ surface-like`)
- `surface_normal n_i`: local normal from weighted covariance eigendecomposition

Given neighbors `j` of atom `i`:

- `u_ij = x_j - x_i`, `r_ij = ||u_ij||`
- neighbor weight `w_ij = exp(-(r_ij / r0)^p)`
- local covariance `Sigma_i = sum_j w_ij * u_ij u_ij^T`
- `n_i`: eigenvector of smallest eigenvalue of `Sigma_i`

`surface_degree` combines:
- smooth coordination deficit
- local anisotropy proxy from covariance eigenvalues

### 2. Anisotropic Surface-Aware Gating in Message Passing

For each edge `(i, j)`, we construct anisotropic terms using

- `c_ij = hat(u_ij) . n_i`
- `c_ij^2` (sign-invariant orientation factor)

and apply multiplicative gates:

- atom message gate: `1 + lambda1 * u_i * u_j + lambda2 * c_ij^2`
- bond/edge update gate: `1 + lambda3 * u_i + lambda4 * c_ij^2`

The same idea is propagated into line-graph (3-body) updates for CHGNet angular interactions.

### 3. Adaptive Receptive Field (Surface-Selective)

We use a coordination-aware effective cutoff:

`r_cut(i,j) = r0 * (1 + alpha_cut * min(u_i, u_j))`

implemented with a smooth sigmoid gate so the model remains fully differentiable.

### 4. Backward Compatibility

All surface-aware behavior is opt-in.

Use:

```python
from matgl.models import CHGNet

model = CHGNet(
    ...,
    surface_aware=True,
    adaptive_receptive_field=True,
)
```

When `surface_aware=False`, behavior follows the original CHGNet path.

## Key Files

- CHGNet surface-aware logic:
  - `src/matgl/models/_chgnet.py`
- Surface-aware gating in graph conv blocks:
  - `src/matgl/layers/_graph_convolution_dgl.py`

## Environment Notes (DGL Backend for CHGNet)

CHGNet in this fork uses DGL backend.

Recommended setup helpers:
- `dev/install_chgnet_cuda_env.sh`
- `dev/chgnet_env_check.py`

Example:

```bash
source .venv/bin/activate
export MATGL_BACKEND=DGL
export MATGL_CACHE=$(pwd)/.matgl_cache
python dev/chgnet_env_check.py --device cuda
```

## Training Guide

See [TRAINING.md](TRAINING.md) for:
- validated environment versions
- stable OC22 finetuning command
- smoke-run command
- known stability settings

## Checkpoints (To Be Updated)

We will publish our latest surface-aware CHGNet checkpoints here:

- [ ] `TBD`: pretrained surface-aware CHGNet checkpoint
- [ ] `TBD`: training config and metrics table
- [ ] `TBD`: inference/demo script for slab systems
