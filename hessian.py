"""Curvature and gradient-noise diagnostics.

Operator convention: every curvature quantity refers to the *symmetric
preconditioned GGN*

    H_hat = P^{-1/2} (J^T J / B) P^{-1/2},

where P^{-1/2} is the same sqrt Adam preconditioner (peak lr and muP
multipliers included) used for the Buvar measurement, and J is the Jacobian
of the model outputs on a fixed held-out batch.  With this convention the
eigenvalues h_i are per-unit-schedule contraction rates: the deterministic
edge of stability sits at s_t * h_i = 2, and the excess-loss response of
mode i is amplified by 2 / (2 - s_t * h_i).

The eigenvectors v_i live in the same coordinates as the preconditioned
per-batch gradients u_b = P^{-1/2} g_b, so noise powers along them are
plain projections Var_b(v_i^T u_b).
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax import tree_util as tu
from jax.tree_util import tree_map

from utils import tree_product, tree_add


# ---------------------------------------------------------------------------
# pytree helpers (single vectors and (k, ...)-stacked vector bundles)
# ---------------------------------------------------------------------------

def tree_normalize(v, eps=1e-30):
    norm = jnp.sqrt(tree_product(v, v))
    return tree_map(lambda x: x / (norm + eps), v)


def tree_rademacher(params, key):
    leaves, treedef = tu.tree_flatten(params)
    keys = jax.random.split(key, len(leaves))
    new = [jax.random.rademacher(k, l.shape, dtype=l.dtype)
           for k, l in zip(keys, leaves)]
    return tu.tree_unflatten(treedef, new)


def stacked_zeros(params, k):
    return tree_map(lambda l: jnp.zeros((k,) + l.shape, l.dtype), params)


def stacked_random(params, key, k):
    leaves, treedef = tu.tree_flatten(params)
    keys = jax.random.split(key, len(leaves))
    new = [jax.random.normal(kk, (k,) + l.shape, l.dtype)
           for kk, l in zip(keys, leaves)]
    return tu.tree_unflatten(treedef, new)


def get_slice(stacked, i):
    return tree_map(lambda l: l[i], stacked)


def update_slice(stacked, i, new):
    return tree_map(lambda l, n: l.at[i].set(n), stacked, new)


def stacked_product(stacked, v):
    """[k] array of inner products between each stacked vector and tree v."""
    prods = tree_map(
        lambda s, x: jnp.sum(s.reshape(s.shape[0], -1) * x.reshape(1, -1),
                             axis=1),
        stacked, v)
    return sum(tu.tree_leaves(prods))


def orth_to_first(v, stacked, count):
    """Project out of v the span of the first `count` stacked vectors."""
    coeffs = stacked_product(stacked, v)
    coeffs = jnp.where(jnp.arange(coeffs.shape[0]) < count, coeffs, 0.0)
    return tree_map(
        lambda x, s: x - jnp.tensordot(coeffs, s, axes=(0, 0)), v, stacked)


# ---------------------------------------------------------------------------
# deflated, warm-startable power iteration
# ---------------------------------------------------------------------------

def _power_iter_one(params, opt_state, v0, stacked, count, *, vp_fn,
                    max_iter, tol):
    def body(state):
        it, v, lam_prev, _ = state
        hv = vp_fn(params, v, opt_state)
        # twice is enough for numerically stable deflation
        hv = orth_to_first(hv, stacked, count)
        hv = orth_to_first(hv, stacked, count)
        lam = tree_product(hv, v)
        v_new = tree_normalize(hv)
        rel = jnp.abs(lam - lam_prev) / (jnp.abs(lam_prev) + 1e-12)
        done = (rel < tol) | (it >= max_iter - 1)
        return it + 1, v_new, lam, done

    it, v, lam, _ = lax.while_loop(
        lambda s: ~s[3], body,
        (jnp.int32(0), v0, jnp.float32(0.0), jnp.bool_(False)))
    return it, v, lam


@partial(jax.jit, static_argnames=("vp_fn", "k", "max_iter", "tol"))
def power_iter_topk(params, opt_state, init_vecs, key, *, vp_fn, k,
                    max_iter=150, tol=1e-4):
    """Top-k eigenpairs of the operator vp_fn by deflated power iteration.

    init_vecs is a (k, ...)-stacked pytree of starting vectors (pass the
    previous call's eigenvectors to warm-start; zeros fall back to random).
    Returns (eigvals [k], eigvecs stacked, iters [k]).
    """
    noise = stacked_random(params, key, k)

    def scan_body(carry, i):
        eigvals, eigvecs, iters = carry
        v0 = get_slice(init_vecs, i)
        v0 = tree_map(lambda x: jnp.where(jnp.isfinite(x), x, 0.0), v0)
        v0 = tree_add(v0, tree_normalize(get_slice(noise, i)), alpha=1e-3)
        v0 = orth_to_first(v0, eigvecs, i)
        v0 = tree_normalize(v0)
        it, v, lam = _power_iter_one(params, opt_state, v0, eigvecs, i,
                                     vp_fn=vp_fn, max_iter=max_iter, tol=tol)
        return (eigvals.at[i].set(lam), update_slice(eigvecs, i, v),
                iters.at[i].set(it)), None

    init = (jnp.zeros(k, dtype=jnp.float32), stacked_zeros(params, k),
            jnp.zeros(k, dtype=jnp.int32))
    (eigvals, eigvecs, iters), _ = lax.scan(scan_body, init, jnp.arange(k))
    # near-degenerate pairs can converge out of order; sort descending
    order = jnp.argsort(-eigvals)
    return (eigvals[order], tree_map(lambda l: l[order], eigvecs),
            iters[order])


@partial(jax.jit, static_argnames=("vp_fn", "probes"))
def hutchinson_trace(params, opt_state, key, *, vp_fn, probes=8):
    keys = jax.random.split(key, probes)
    tr = jnp.float32(0.0)
    for i in range(probes):
        v = tree_rademacher(params, keys[i])
        tr += tree_product(v, vp_fn(params, v, opt_state))
    return tr / probes


# ---------------------------------------------------------------------------
# gradient-noise probe along the eigenvectors and a fixed direction
# ---------------------------------------------------------------------------

def noise_probe_fn(params, dataset, opt_state, vecs, dir_vec, *,
                   sqrt_P_inv, loss_and_grad_fn):
    """Scan held-out batches; project preconditioned per-batch gradients
    u_b = P^{-1/2} g_b onto the k stacked vectors and onto dir_vec.

    Returns (metrics dict, umean tree).  sigma2[i] is the per-batch variance
    of v_i^T u_b (multiply by the batch size for the per-example variance),
    vproj[i] the mean projection v_i^T u_bar; likewise sigma2_dir / dproj
    along dir_vec.  Buvar_h / umean2_h are high-precision versions of the
    standard eval metrics.
    """
    k = tu.tree_leaves(vecs)[0].shape[0]
    carry = dict(
        n=jnp.float32(0.0),
        u_sum=tree_map(jnp.zeros_like, params),
        u2_sum=jnp.float32(0.0),
        loss_sum=jnp.float32(0.0),
        proj_sum=jnp.zeros(k, dtype=jnp.float32),
        proj2_sum=jnp.zeros(k, dtype=jnp.float32),
        pd_sum=jnp.float32(0.0),
        pd2_sum=jnp.float32(0.0),
    )

    def body(carry, batch):
        loss, grads = loss_and_grad_fn(params, batch)
        u = sqrt_P_inv(grads, opt_state)
        pr = stacked_product(vecs, u)
        pd = tree_product(dir_vec, u)
        new = dict(
            n=carry['n'] + 1.0,
            u_sum=tree_add(carry['u_sum'], u),
            u2_sum=carry['u2_sum'] + tree_product(u, u),
            loss_sum=carry['loss_sum'] + loss,
            proj_sum=carry['proj_sum'] + pr,
            proj2_sum=carry['proj2_sum'] + pr ** 2,
            pd_sum=carry['pd_sum'] + pd,
            pd2_sum=carry['pd2_sum'] + pd * pd,
        )
        return new, None

    carry, _ = lax.scan(body, carry, dataset)
    n = carry['n']
    bessel = n / (n - 1.0)
    umean = tree_map(lambda x: x / n, carry['u_sum'])
    umean2 = tree_product(umean, umean)
    proj_mean = carry['proj_sum'] / n
    pd_mean = carry['pd_sum'] / n
    metrics = dict(
        Buvar_h=(carry['u2_sum'] / n - umean2) * bessel,
        umean2_h=umean2,
        loss_h=carry['loss_sum'] / n,
        sigma2=(carry['proj2_sum'] / n - proj_mean ** 2) * bessel,
        vproj=proj_mean,
        sigma2_dir=(carry['pd2_sum'] / n - pd_mean ** 2) * bessel,
        dproj=pd_mean,
    )
    return metrics, umean


def hessian_eval_schedule(num_train_steps, schedule, decay_frac, n_pre,
                          n_post):
    """Dense curvature-eval steps: log-spaced early plus linear coverage of
    the pre-decay window, then linear coverage of the decay window."""
    import numpy as np
    T = num_train_steps
    if schedule == 'const' or decay_frac >= 1.0:
        # no interior decay boundary: log-spaced half, linear half over the run
        pts = set(np.geomspace(1, T - 1, n_pre).astype(int))
        pts |= set(np.linspace(0, T - 1, n_post).astype(int))
    else:
        anchor = max(2, int(round((1.0 - decay_frac) * T)))
        pts = set(np.geomspace(1, anchor, n_pre // 2).astype(int))
        pts |= set(np.linspace(0, anchor, n_pre - n_pre // 2).astype(int))
        pts |= set(np.linspace(anchor, T - 1, n_post).astype(int))
    pts |= {1, T - 1}
    # step 0 is excluded: the Adam second-moment state is empty there and the
    # preconditioner is undefined (0/0 bias correction)
    pts.discard(0)
    return {int(p) for p in pts}
