"""Dual kernel approximation with Hermite expansion and Monte-Carlo method."""

import jax
from jax.config import config

config.update("jax_enable_x64", True)

import time
import numpy as np
import argparse

from ntk_activations.polynomials import hermite_polfit, hermite_coeffs_relu, hermite2mono
from ntk_activations.dual_kernels import dual_kernel_exact, dual_kernel_poly, dual_kernel_empirical, get_act


def test_single_layer_nngp_kernel_approx():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--n_train", type=int, default=1000)
  parser.add_argument("--n_dim", type=int, default=256)
  parser.add_argument("--act",
                      type=str,
                      default='gelu',
                      choices=['gelu', 'relu', 'sin', 'erf', 'gaussian', 'abs'])
  parser.add_argument("--num_iters", type=int, default=10)
  args = parser.parse_args()

  for name_, value_ in args.__dict__.items():
    print(f"{name_:<15} : {value_}")

  degrees = np.arange(1, 20)
  num_hiddens = [2**i for i in range(5, 17)]

  errors = {}
  errors['hermite'] = np.zeros((args.num_iters, len(degrees)))
  errors['mc'] = np.zeros((args.num_iters, len(num_hiddens)))

  times = {}
  times['exact'] = np.zeros(args.num_iters)
  times['hermite'] = np.zeros((args.num_iters, len(degrees)))
  times['mc'] = np.zeros((args.num_iters, len(num_hiddens)))

  tic0 = time.time()
  func = lambda x_: get_act(x_, args.act)
  print("coefficients computations...")
  coeffs_all = {'taylor': {}, 'cheby': {}, 'hermite': {}}
  if args.act not in ['relu']:
    coeffs_hermite_all = hermite_polfit(func, max(degrees), False)

  for deg in degrees:
    if args.act == 'relu':
      coeffs_all['hermite'][deg] = hermite_coeffs_relu(deg, True)
    else:
      coeffs_all['hermite'][deg] = hermite2mono(coeffs_hermite_all[:deg + 1])

  for i in range(args.num_iters):
    print(f"\n[{i} / {args.num_iters}]")

    key1 = jax.random.PRNGKey(args.seed + 23 * i)

    n = args.n_train
    d = args.n_dim
    x = jax.random.normal(key1, (n, d)) / d**0.5

    print(f"input shape : {x.shape} | checksum : {x.sum()}")

    # 1. compute the exact dual kernel.
    tic = time.time()
    k_exact = dual_kernel_exact(x, args.act)
    tim_exact = time.time() - tic
    print(f"{'Exact time':<15} : {tim_exact:.3f} sec")

    k_exact_norm = np.linalg.norm(k_exact, 'fro')

    checksum = k_exact.sum()
    print(f"k_exact check sum : {checksum:.4f}")

    # 2. estimate the dual kernel via Hermite series expansion
    j = 0
    
    for deg in degrees:
      coeffs = coeffs_all['hermite'][deg]

      tic1 = time.time()
      k_poly = dual_kernel_poly(coeffs, x, 1)
      times['hermite'][i, j] = time.time() - tic1

      rel_err = np.linalg.norm(k_exact - k_poly, 'fro') / k_exact_norm

      errors['hermite'][i, j] = rel_err
      j += 1

    # 3. estimate by Monte-Carlo method
    j = 0
    for m in num_hiddens:
      key2 = jax.random.PRNGKey(args.seed + 19 * i)

      tic2 = time.time()
      k_mc = dual_kernel_empirical(x, key2, m, args.act)
      times['mc'][i, j] = time.time() - tic2

      rel_err2 = np.linalg.norm(k_exact - k_mc, 'fro') / k_exact_norm

      errors['mc'][i, j] = rel_err2
      j += 1

    print(
        f"{i} / {args.num_iters} | time_elapsed : {time.time() - tic0:.4f} sec")
    print(
        f'err_hermt: {", ".join([f"{a:.2e}" for a in np.mean(errors["hermite"][:i+1], axis=0)])}'
    )
    print(
        f'err_mc   : {", ".join([f"{a:.2e}" for a in np.mean(errors["mc"][:i+1], axis=0)])}'
    )
    print("-" * 140)
    print(
        f'tim_hermt: {", ".join([f"{a:.2e}" for a in np.mean(times["hermite"][:i+1], axis=0)])}'
    )
    print(
        f'tim_mc   : {", ".join([f"{a:.2e}" for a in np.mean(times["mc"][:i+1], axis=0)])}'
    )
    print("-" * 140)
    print(f"time_elapsed : {time.time() - tic0:.4f} sec")


if __name__ == "__main__":
  test_single_layer_nngp_kernel_approx()
