from jax.scipy.special import gammaln
import scipy.integrate as integrate
import numpy as onp


def hermite_polval_single(t, degree):
  if degree == 0:
    return onp.ones_like(t)
  elif degree == 1:
    return t
  else:
    y0 = onp.ones_like(t)
    y1 = t
    for i in range(2, degree + 1):
      y2 = t * y1 - (i - 1) * y0
      y0 = y1
      y1 = y2
    return y2


def hermite_polval(t, coeffs):
  degree = len(coeffs) - 1

  if degree == 0:
    return onp.ones_like(t) * coeffs[0]

  elif degree == 1:
    return onp.ones_like(t) * coeffs[0] + t * coeffs[1]

  else:
    y0 = onp.ones_like(t)
    y1 = t
    y = y0 * coeffs[0] + y1 * coeffs[1]
    for i in range(2, degree + 1):
      y2 = t * y1 - (i - 1) * y0
      y += coeffs[i] * y2
      y0 = y1
      y1 = y2
    return y


def hermite_polfit(func, degree, convert_to_mono=False):
  weight_func = lambda x: onp.exp(-x**2 / 2.)
  coeffs = []
  for i in range(degree + 1):
    ci = integrate.quad(
        lambda x: func(x) * weight_func(x) * hermite_polval_single(x, i),
        -onp.inf, onp.inf)[0]
    ci /= onp.sqrt(2 * onp.pi) * onp.exp(gammaln(i + 1))
    coeffs.append(ci)
  if not convert_to_mono:
    return onp.array(coeffs)
  else:
    return hermite2mono(onp.array(coeffs))


def hermite2mono(coeffs):
  degree = len(coeffs) - 1
  transition_mtx = onp.zeros((degree + 1, degree + 1))
  transition_mtx[0, 0] = 1.

  for i in range(degree):
    transition_mtx[i + 1, 1:] += transition_mtx[i, :-1]
    if i >= 1:
      transition_mtx[i + 1, :] -= i * transition_mtx[i - 1, :]

  return (coeffs.reshape(1, -1) @ transition_mtx)[-1,]


def hermite_coeffs_relu(degree, convert_to_mono=False):
  two_pi_sqrt = onp.sqrt(2 * onp.pi)
  coeffs = []
  for i in range(degree + 1):
    if i == 0:
      coeffs.append(1 / two_pi_sqrt)
    elif i == 1:
      coeffs.append(.5)
    elif i % 2 == 1:
      coeffs.append(0.)
    else:
      double_fac_i_minus_2 = onp.exp(
          gammaln((i - 2) / 2 + 1) + (i - 2) / 2 * onp.log(2))
      ci = i * (i - 1) * double_fac_i_minus_2 * (-1)**(i / 2 - 1)
      coeffs.append(1 / ci / two_pi_sqrt)

  if not convert_to_mono:
    return onp.array(coeffs)
  else:
    return hermite2mono(onp.array(coeffs))
