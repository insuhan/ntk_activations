import jax
from jax.scipy.special import gammaln
import jax.numpy as jnp
from jax import random


def get_act(x, name):
  if name == 'relu':
    return jax.nn.relu(jnp.asarray(x))._value
  elif name == 'relu6':
    return jax.nn.relu6(jnp.asarray(x))._value
  elif name == 'sigmoid':
    return jax.nn.sigmoid(jnp.asarray(x))._value
  elif name == 'softplus':
    return jax.nn.softplus(jnp.asarray(x))._value
  elif name == 'soft_sign':
    return jax.nn.soft_sign(jnp.asarray(x))._value
  elif name == 'silu':
    return jax.nn.silu(jnp.asarray(x))._value
  elif name == 'swish':
    return jax.nn.swish(jnp.asarray(x))._value
  elif name == 'log_sigmoid':
    return jax.nn.log_sigmoid(jnp.asarray(x))._value
  elif name == 'leaky_relu':
    return jax.nn.log_sigmoid(jnp.asarray(x))._value
  elif name == 'hard_sigmoid':
    return jax.nn.hard_sigmoid(jnp.asarray(x))._value
  elif name == 'hard_silu':
    return jax.nn.hard_silu(jnp.asarray(x))._value
  elif name == 'hard_swish':
    return jax.nn.hard_swish(jnp.asarray(x))._value
  elif name == 'hard_tanh':
    return jax.nn.hard_tanh(jnp.asarray(x))._value
  elif name == 'elu':
    return jax.nn.elu(jnp.asarray(x))._value
  elif name == 'celu':
    return jax.nn.celu(jnp.asarray(x))._value
  elif name == 'selu':
    return jax.nn.selu(jnp.asarray(x))._value
  elif name == 'gelu':
    return jax.nn.gelu(jnp.asarray(x), False)._value
  elif name == 'glu':
    return jax.nn.glu(jnp.asarray(x))._value
  elif name == 'sin':
    return jnp.sin(jnp.asarray(x))._value
  elif name == 'cos':
    return jnp.cos(jnp.asarray(x))._value
  elif name == 'erf':
    return jax.scipy.special.erf(jnp.asarray(x))._value
  elif name == 'gaussian':
    return jnp.exp(-0.5 * jnp.asarray(x)**2)
  elif name == 'abs':
    return jnp.abs(x)
  else:
    raise NotImplementedError


def dual_kernel_exact(x: jnp.ndarray, act: str, is_x_matrix: bool = False):

  if is_x_matrix:
    xxt = x
    normsq = x.diag()
  else:
    xxt = x @ x.T
    normsq = jnp.linalg.norm(x, axis=-1)**2

  if act == 'gelu':
    cos_ = (1 / jnp.sqrt(normsq)[None, :]) * (xxt / jnp.sqrt(normsq)[:, None])
    sin_sq = 1 - jnp.clip(cos_**2, -1, 1)

    normsq_mtx = jnp.outer(normsq, normsq)
    normsq_plus_1 = jnp.outer(1 + normsq, 1 + normsq)
    a = normsq[:, None] + normsq[None, :] + normsq_mtx * sin_sq

    a1 = (cos_**2 + 1 + a) / normsq_plus_1 / jnp.sqrt(1 + a)
    a2 = xxt / normsq_mtx * jnp.arctan(xxt / jnp.sqrt(1 + a))

    return xxt / 4 + (normsq_mtx / 2 / jnp.pi) * (a1 + a2)

  elif act == 'sin':
    return _mul_diag_mtx(jnp.exp(-normsq / 2)[:, None], jnp.sinh(xxt))

  elif act == 'cos':
    return _mul_diag_mtx(jnp.exp(-normsq / 2)[:, None], jnp.cosh(xxt))

  elif act == 'erf':
    return (2 / jnp.pi) * jnp.arcsin(
        _mul_diag_mtx(1 / jnp.sqrt(1 + 2 * normsq), 2 * xxt))

  elif act == 'gaussian':  # exp(-0.5 * t**2)
    return 1 / jnp.sqrt(jnp.outer(1 + normsq, 1 + normsq) - xxt**2)

  elif act == 'relu':
    cos_ = (1 / jnp.sqrt(normsq)[None, :]) * (xxt / jnp.sqrt(normsq)[:, None])
    angle = jnp.arccos(jnp.clip(cos_, -1, 1))
    angle = (jnp.pi - angle) / jnp.pi * cos_ + jnp.sqrt(
        jnp.clip(1 - cos_**2, a_min=0)) / jnp.pi
    return _mul_diag_mtx(normsq**0.5, angle) / 2.

  elif act == 'abs':
    cos_ = (1 / jnp.sqrt(normsq)[None, :]) * (xxt / jnp.sqrt(normsq)[:, None])
    angle = jnp.arccos(jnp.clip(cos_, -1, 1))
    angle2 = (jnp.pi - angle) / jnp.pi * cos_ + jnp.sqrt(
        jnp.clip(1 - cos_**2, a_min=0)) / jnp.pi
    return jnp.outer(normsq**0.5, normsq**0.5) * (2 * angle2 - cos_)

  else:
    raise NotImplementedError


def _mul_diag_mtx(diag_: jnp.ndarray, mtx_: jnp.ndarray):
  return jnp.outer(diag_, diag_) * mtx_


def dual_kernel_poly(coeffs: jnp.ndarray, x: jnp.ndarray, a: float = 1.0):
  xnorms = jnp.linalg.norm(x, axis=-1)[:, None] / a
  log_xnorms = jnp.log(xnorms)
  x_ = x / xnorms
  xxt = x_ @ x_.T
  deg = len(coeffs) - 1

  for l in range(deg + 1):
    j_all = jnp.arange((deg - l) // 2 + 1)
    log_factors = gammaln(2 * j_all + l + 1) - gammaln(
        j_all + 1) - j_all * jnp.log(2) - 0.5 * gammaln(l + 1)
    radial_l = jnp.dot(jnp.exp(log_factors + (2 * j_all + l) * log_xnorms),
                       coeffs[l::2])

    if l == 0:
      out = jnp.outer(radial_l, radial_l)
      xxt_l = xxt
    elif l == deg:
      return out
    else:
      out += jnp.outer(radial_l, radial_l) * xxt_l
      xxt_l = xxt_l * xxt


def dual_kernel_empirical(x: jnp.ndarray,
                          rng: random.KeyArray,
                          num_hiddens: int = 1024,
                          act: str = 'gelu'):
  w = jax.random.normal(rng, (x.shape[-1], num_hiddens))
  z = get_act(x @ w, act) / num_hiddens**0.5
  return z @ z.T
