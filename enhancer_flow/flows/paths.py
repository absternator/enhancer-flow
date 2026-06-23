import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from ..data import VOCAB_SIZE


EPS = 1e-6 # clip distance from simplex verticies/boundary

def sample_dirichlet_path(
    key: Array,
    x1: Int[Array, "L"],
    t: Float[Array, "B"],
    *,
    alpha_scale: float = 1.0,
    num_classes: int = VOCAB_SIZE,
) -> Float[Array, "B L K"]:
    """Sample ``x_t ~ Dir(1 + t*alpha_scale*e_{x1})`` per position.

    Args:
        key: PRNG key.
        x1: clean nucleotide ids ``(B, L)`` in ``0..K-1``.
        t: time per example ``(B,)`` in ``[0, 1]``.
        alpha_scale: growth rate of the concentration toward the true vertex.
            Larger -> paths concentrate faster. Paper uses a schedule; a constant
            is a fine starting point and keeps the loss well-behaved.
        num_classes: K (4 for DNA).

    Returns:
        ``x_t`` on the simplex, shape ``(B, L, K)``, rows summing to 1.
    """
    onehot = jax.nn.one_hot(x1, num_classes=num_classes) # (B, L, K)
    # alpha = 1 + t * alpha_scale * e_{x1}; broadcast t over L and K.
    t_blk = t[..., None, None] # (B, 1, 1)
    alpha = 1.0 + t_blk * alpha_scale * onehot # (B, L, K)
    # Dirichlet(alpha) sample. jax.random.dirichlet wants alpha on the last axis.
    xt = jax.random.dirichlet(key, alpha) # (B, L, K)
    # Clip away from exact 0/1 for downstream log-space stability.
    xt = jnp.clip(xt, EPS, 1.0 - EPS)
    xt = xt / xt.sum(axis=-1, keepdims=True) # renormalize to sum to 1
    return xt

def denoising_ce_loss(
    logits: Float[Array, "B L K"],
    x1: Int[Array, "B L"],
    *,
    mask: Int[Array, "B L"] | None = None,
) -> Float[Array, ""]:
    """Cross-entropy between predicted clean-class logits and true ids.

    This is the entire Phase-1 training objective. Stable and simple — no score
    matching required at train time; the score/velocity is recovered at sampling
    from the predicted distribution.

    Args:
        logits: network output ``(B, L, K)``.
        x1: clean ids ``(B, L)``.
        mask: optional ``(B, L)`` of 1 for real positions, 0 for pads.
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    onehot = jax.nn.one_hot(x1, num_classes=logits.shape[-1], dtype=logits.dtype)

    cross_entropy = -(onehot * log_probs).sum(axis=-1) # (B, L)
    if mask is not None:
        cross_entropy = cross_entropy * mask
        denom = jnp.maximum(mask.sum(), 1.0)
        return cross_entropy.sum() / denom
    return cross_entropy.mean()

def vector_field_from_logits(
    logits: Float[Array, "B L K"],
    xt: Float[Array, "B L K"],
    t: Float[Array, "B"],
    *,
    alpha_scale: float = 1.0,
    t_max: float = 100.0,
) -> Float[Array, "B L K"]:
    """Closed-form Dirichlet flow velocity ``v(x_t, t)`` from predicted clean logits.

    The network predicts the clean-class posterior ``x_hat_1 = p(x_1 | x_t, t)`` per
    position. The marginal velocity is that posterior-weighted mixture of the
    per-vertex conditional flows:

        v(x, t) = sum_i  x_hat_1[i] * [C(x_i, t) / (1 - x_i)] * (e_i - x)

    Args:
        logits: predicted clean-class logits ``(B, L, K)``.
        xt: current simplex state ``(B, L, K)``.
        t: time ``(B,)`` in ``[0, 1]``.
        alpha_scale: growth rate of the concentration toward the true vertex.
            Larger -> paths concentrate faster. Paper uses a schedule; a constant
            is a fine starting point and keeps the loss well-behaved.
        t_max: maximum time for scaling the velocity. The paper uses 100.

    Returns:
        velocity ``v(x_t, t)`` of shape ``(B, L, K)``.
    """
    probs = jax.nn.softmax(logits, axis=-1) # x_hat_1, (B,L,K)
    c_factor = _dirichlet_c_factor(xt, t, t_max=t_max, num_classes=logits.shape[-1]) # scalar coord velocity (B, L, K)
    scaled_c = c_factor / jnp.clip(1.0 - xt, EPS, None) # [C_i/(1-x_i)] -  (B, L, K)
    # weighted mixture of the per-vertex conditional flows
    weighted = probs * scaled_c # (B, L, K)
    # Mixture sum_i probs_i * scale_i * (e_i - x), assembled without the
    # (B,L,K,K) tensor:
    #   diagonal part:  sum_i probs_i scale_i e_i  = probs * scale
    #   shared -x part: -(sum_i probs_i scale_i) x = -(probs*scale).sum(-1) * x
    v = weighted - weighted.sum(axis=-1, keepdims=True) * xt # (B, L, K)
    # re-center to kill any residual numerical drift off the tangent space
    v = v - v.mean(axis=-1, keepdims=True)
    return v



def _dirichlet_c_factor(
    xt: Float[Array, "B L K"],
    t: Float[Array, "B"],
    *,
    num_classes: int = VOCAB_SIZE,
    t_max: float = 100.0,
) -> Float[Array, "B L K"]:
    """Per-class C-factor for the conditional Dirichlet flow (Stark et al. 2024).

    The conditional probability path toward vertex ``i`` is
    ``p_a(x | e_i) = Dir(alpha)`` with ``alpha_i = a`` and ``alpha_j = 1`` for
    ``j != i``, where ``a`` grows from 1 to a large terminal concentration. The
    marginal of coordinate ``x_i`` under this path is ``Beta(a, K - 1)``, whose
    mean ``a / (a + K - 1)`` only approaches 1 (the vertex) as ``a`` grows large —
    so the sampler's normalized time ``t in [0, 1]`` must be reparametrized to a
    growing concentration. We use the linear schedule

        a(t) = 1 + t * (t_max - 1)        ->   a(0) = 1 (uniform prior),
                                                a(1) = t_max (near the vertex)

    The conditional vector field that transports the prior along this path is

        u(x | e_i) = C(x_i, t) * (e_i - x)

    pointing from the current point toward vertex ``i`` (automatically tangent, and
    vanishing at the vertex and the opposite face — the paper's design property).
    The scalar C-factor follows the 1-D continuity equation for the marginal
    ``Beta(a, K-1)``:

        C(x_i, t) = - [ d/dt I_{x_i}(a, K-1) ] / Beta_pdf(x_i; a, K-1)
                  = - (da/dt) * [ d/da I_{x_i}(a, K-1) ] / Beta_pdf(x_i; a, K-1)

    where ``I`` is the regularized incomplete beta (the Beta CDF). ``da/dt =
    t_max - 1`` is the chain-rule factor that was missing before and is what makes
    the field actually reach the vertex by ``t = 1``. JAX cannot autodiff
    ``betainc`` through its shape parameters, so ``d/da I`` is a central finite
    difference in ``a`` (the C-factor is smooth); the whole field is validated
    end-to-end against the conditional-flow property (prior -> correct vertex) in
    the tests.

    ``t_max`` sets how sharply the terminal distribution concentrates on the
    vertex; ~100 gives true-class mass ~0.97 at ``t=1``, plenty for argmax decoding.
    Returns the per-class C-factor ``(B, L, K)``; the caller multiplies by
    ``(e_i - x)`` and mixes over the predicted clean distribution.
    """
    da_dt = t_max - 1.0
    a = (1.0 + t[..., None, None] * da_dt) # Beta first shape parameter a(t) - (B, 1, 1)
    b = float(num_classes - 1) # Beta second shape parameter
    xt_safe = jnp.clip(xt, EPS, 1.0 - EPS) # avoid log(0) in Beta_pdf

    # d/da of the regularized incomplete beta I_x(a, b) via central finite
    # difference (betainc is not autodiff-able through its shape parameters).
    h = 1e-3
    a_hi = a + h
    a_lo = jnp.clip(a - h, 1.0 + EPS, None) # avoid a < 1
    I_hi = jax.scipy.special.betainc(a_hi, b, xt_safe)
    I_lo = jax.scipy.special.betainc(a_lo, b, xt_safe)
    dI_da = (I_hi - I_lo) / (a_hi - a_lo)

    # Beta pdf: x^(a-1) (1-x)^(b-1) / B(a,b); compute in log space for stability.
    log_pdf = (a - 1.0) * jnp.log(xt_safe) + (b - 1.0) * jnp.log1p(-xt_safe) - _log_beta(a, b)
    pdf = jnp.exp(log_pdf)

    return -da_dt * dI_da / jnp.clip(pdf, EPS, None)


def _log_beta(a: Array, b: float) -> Array:
    """log B(a, b) = lgamma(a) + lgamma(b) - lgamma(a + b)."""
    lg = jax.scipy.special.gammaln
    return lg(a) + lg(b) - lg(a + b)