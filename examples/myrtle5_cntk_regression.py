from tqdm import tqdm
import time
import argparse
import numpy as onp
import tensorflow_datasets as tfds
import jax
from jax import jit
import jax.numpy as np
from jax import random

from ntk_activations.stax_extensions_features import ExpNormalizedFeatures, ConvFeatures, AvgPoolFeatures, FlattenFeatures, DenseFeatures, serial

layer_factor = {5: [2, 1, 1], 7: [2, 2, 2], 10: [3, 3, 3]}


def MyrtleNetworkFeatures(act,
                          depth=5,
                          W_std=np.sqrt(2.0),
                          width=1,
                          **relu_args):

  act_func = ExpNormalizedFeatures

  layers = []
  layers += [
      ConvFeatures(width, filter_shape=(3, 3), W_std=W_std),
      act_func(**relu_args)
  ] * layer_factor[depth][0]
  layers += [AvgPoolFeatures((2, 2), strides=(2, 2))]
  layers += [
      ConvFeatures(width, filter_shape=(3, 3), W_std=W_std),
      act_func(**relu_args)
  ] * layer_factor[depth][1]
  layers += [AvgPoolFeatures((2, 2), strides=(2, 2))]
  layers += [
      ConvFeatures(width, filter_shape=(3, 3), W_std=W_std),
      act_func(**relu_args)
  ] * layer_factor[depth][2]
  layers += [AvgPoolFeatures((2, 2), strides=(2, 2))] * 3
  layers += [FlattenFeatures(), DenseFeatures(1, W_std)]
  return serial(*layers)


def get_preprocess_op(x_train,
                      layer_norm=True,
                      zca_reg=1e-5,
                      zca_reg_absolute_scale=False,
                      on_cpu=False):
  """ZCA preprocessing function."""
  whitening_transform = _get_whitening_transform(x_train, layer_norm, zca_reg,
                                                 zca_reg_absolute_scale, on_cpu)
  per_channel_mean = np.mean(x_train, axis=(0, 1, 2))
  per_channel_std = np.std(x_train, axis=(0, 1, 2))

  def _preprocess_op(images):
    images = (images - per_channel_mean) / per_channel_std

    orig_shape = images.shape
    images = images.reshape(orig_shape[0], -1)
    if layer_norm:
      # Zero mean every feature
      images = images - np.mean(images, axis=1)[:, np.newaxis]
      # Normalize
      image_norms = np.linalg.norm(images, axis=1)
      # Make features unit norm
      images = images / image_norms[:, np.newaxis]

    images = (images).dot(whitening_transform)
    images = images.reshape(orig_shape)
    return images

  return _preprocess_op


def _get_whitening_transform(x_train,
                             layer_norm=True,
                             zca_reg=1e-5,
                             zca_reg_absolute_scale=False,
                             on_cpu=False):
  """Returns 2D matrix that performs whitening transform.

  Whitening transform is a (d,d) matrix (d = number of features) which acts on
  the right of a (n, d) batch of flattened data.
  """
  orig_train_shape = x_train.shape
  x_train = x_train.reshape(orig_train_shape[0], -1).astype('float64')
  if on_cpu:
    x_train = jax.device_put(x_train, jax.devices('cpu')[0])

  n_train = x_train.shape[0]
  if layer_norm:
    print('Performing layer norm preprocessing.')
    # Zero mean every feature
    x_train = x_train - np.mean(x_train, axis=1)[:, np.newaxis]
    # Normalize
    train_norms = np.linalg.norm(x_train, axis=1)
    # Make features unit norm
    x_train = x_train / train_norms[:, np.newaxis]

  print(f'Performing zca whitening preprocessing with reg: {zca_reg}')
  cov = 1.0 / n_train * x_train.T.dot(x_train)
  if zca_reg_absolute_scale:
    reg_amount = zca_reg
  else:
    reg_amount = zca_reg * np.trace(cov) / cov.shape[0]
  print(f'Raw zca regularization strength: {reg_amount}')

  u, s, _ = np.linalg.svd(cov + reg_amount * np.eye(cov.shape[0]))
  inv_sqrt_zca_eigs = s**(-1 / 2)

  # rank control
  if n_train < x_train.shape[1]:
    inv_sqrt_zca_eigs = inv_sqrt_zca_eigs.at[n_train:].set(
        np.ones(inv_sqrt_zca_eigs[n_train:].shape[0]))
  whitening_transform = np.einsum('ij,j,kj->ik',
                                  u,
                                  inv_sqrt_zca_eigs,
                                  u,
                                  optimize=True)
  return whitening_transform


def get_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("--act", type=str, default='normgauss')
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--depth", type=int, default=5)
  parser.add_argument("--degree", type=int, default=8)
  parser.add_argument("--normgauss_a", type=float, default=1.0)
  parser.add_argument("--feat_dim", type=int, default=8192)
  parser.add_argument("--batch_size", type=int, default=1)
  args = parser.parse_args()
  return args


def main():
  args = get_arguments()

  for n_, v_ in args.__dict__.items():
    print(f"{n_:<20} : {v_}")

  ds_train, ds_test = tfds.as_numpy(
      tfds.load('cifar10', split=['train', 'test'], batch_size=-1))

  x_train_ = ds_train['image']
  y_train = ds_train['label']
  x_test_ = ds_test['image']
  y_test = ds_test['label']

  preprocess_op = get_preprocess_op(x_train_, layer_norm=True, zca_reg=0.1)
  tic = time.time()
  x_train = preprocess_op(x_train_)
  x_test = preprocess_op(x_test_)
  print(f"[INFO] preprocessing is done. time: {time.time() - tic:.4f} sec")

  print(f"x_train.shape: {x_train.shape}")
  print(f"x_test.shape: {x_test.shape}")

  y_label = jax.nn.one_hot(y_train, 10)

  key = random.PRNGKey(0)
  key1, key2 = random.split(key)
  depth = args.depth
  act = args.act
  poly_degree = args.degree
  poly_sketch_dim = args.feat_dim
  sketch_dim = args.feat_dim
  x = x_train[:1]

  act = f"normgauss{args.normgauss_a}"
  print(
      f"depth: {depth}, act: {act}, poly_degree: {poly_degree}, poly_sketch_dim: {poly_sketch_dim}, sketch_dim: {sketch_dim}"
  )

  kwarg = {
      'sketch_dim': sketch_dim,
      'poly_degree': poly_degree,
      'poly_sketch_dim': poly_sketch_dim
  }

  kwarg['a'] = args.normgauss_a

  init_fn, feature_fn = MyrtleNetworkFeatures(args.act, depth, **kwarg)
  feature_fn = jit(feature_fn)
  _, feat_fn_inputs = init_fn(key2, x.shape)

  def batch_collect_featues(feature_fn, feat_fn_inputs, x, batch_size):
    nngp_feat = []
    ntk_feat = []
    n_data = x.shape[0]
    num_batches = int(np.ceil(n_data / batch_size))
    for batch_idx in tqdm(range(num_batches)):
      current_x = x[batch_idx * batch_size:(batch_idx + 1) * batch_size]
      feats_complex = feature_fn(current_x, feat_fn_inputs)
      nngp_feat_ = onp.concatenate(
          (feats_complex.nngp_feat.real, feats_complex.nngp_feat.imag), axis=-1)
      ntk_feat_ = onp.concatenate(
          (feats_complex.ntk_feat.real, feats_complex.ntk_feat.imag), axis=-1)
      nngp_feat.append(nngp_feat_)
      ntk_feat.append(ntk_feat_)

    return onp.concatenate(nngp_feat, axis=0), onp.concatenate(ntk_feat, axis=0)

  batch_size = args.batch_size
  tic = time.time()
  nngp_feat_test, ntk_feat_test = batch_collect_featues(feature_fn,
                                                        feat_fn_inputs,
                                                        x_test,
                                                        batch_size=batch_size)
  toc = time.time() - tic
  print(
      f"test feat shape nngp: {nngp_feat_test.shape}, ntk: {ntk_feat_test.shape} compute time: {toc:.2f}s"
  )

  batch_size = args.batch_size
  tic = time.time()
  nngp_feat_train, ntk_feat_train = batch_collect_featues(feature_fn,
                                                          feat_fn_inputs,
                                                          x_train[:50000],
                                                          batch_size=batch_size)
  toc = time.time() - tic
  print(
      f"train feat shape nngp: {nngp_feat_train.shape}, ntk: {ntk_feat_train.shape}  time: {toc:.2f}s"
  )

  # Full cifar inference can't be done on A100
  nngp_feat_train = jax.device_put(nngp_feat_train, jax.devices('cpu')[0])
  nngp_feat_test = jax.device_put(nngp_feat_test, jax.devices('cpu')[0])

  ntk_feat_train = jax.device_put(ntk_feat_train, jax.devices('cpu')[0])
  ntk_feat_test = jax.device_put(ntk_feat_test, jax.devices('cpu')[0])

  epsilons = np.logspace(-10, 2, 20)

  Amat = ntk_feat_train.T @ ntk_feat_train
  bvec = ntk_feat_train.T @ y_label
  print(f"Amat.shape: {Amat.shape}, bvec.shape: {bvec.shape}")
  best_eps = -1
  best_acc = -1
  for epsilon in epsilons:
    pred = ntk_feat_test @ np.linalg.solve(
        Amat + epsilon * np.eye(ntk_feat_train.shape[1]), bvec)
    acc = np.mean(np.argmax(pred, axis=-1) == y_test)
    print(f"{epsilon:.2e}, {acc:.4f}", end='')

    if best_acc < acc:
      best_acc = acc
      best_eps = epsilon
      print("... (best)")
    else:
      print("")

  print(f"best_acc_ntk: {best_acc}, best_eps_ntk: {best_eps}")


if __name__ == "__main__":
  main()
