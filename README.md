# Fast Neural Kernel Embeddings for General Activations

## Install

Clone the package: 
```commandline 
git clone https://github.com/insuhan/ntk_activations.git
cd ntk_activations 
pip install -e .
```

## Usage

To run dual kernel approximations with Hermite expansion for GeLU activation:
```commandline
python examples/dual_kernel_approx.py --act gelu
```
For other activations such as `relu`, `sin`, `gaussian`, `erf`, `abs`, please replace the argument `gelu` with the other one (e.g., `--act erf`).

To run convolutional NTK (CNTK) sketch algorithm for regression with CIFAR-10 dataset:
```command
python examples/myrtle5_cntk_regression.py
```
This approximates CNTK of depth-5 convolutional neural networks (a.k.a. Myrtle-5) by sketching algorithms where dual kernel of its activation corresponds to the normalized Gaussian kernel. A scaling factor of the normalized Gaussian kernel is changed with argument, e.g., `--normgauss_a 0.5` (default is `1`). All modules for NTK features are based on [`neural_tangents`](https://github.com/google/neural-tangents) (see `ntk_activations/stax_extensions_features.py`) and sketching algorithms are implemented in `ntk_activations/sketching.py`.
