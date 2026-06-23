import jax
import jax.numpy as jnp
from enhancer_flow.flows.paths import _dirichlet_c_factor, vector_field_from_logits

B, L, K = 16, 32, 4

def test_cfactor_non_negative():
    """C-factor transports mass toward the vertex over time, so it is
    non-negative everywhere (it vanishes near faces — the design property —
    but is never meaningfully negative)."""
    xt = jax.random.dirichlet(jax.random.key(7), jnp.ones((B, L, K)))
    t = jnp.full((B, ), 0.4)
    c = _dirichlet_c_factor(xt, t, num_classes=K)
    assert float(c.min()) > -1e-4, f"C-factor has negative values: min={float(c.min())}"
    assert jnp.isfinite(c).all()

def test_field_is_tangent():
    xt = jax.random.dirichlet(jax.random.key(0), jnp.ones((B, L, K)))
    logits = jax.random.normal(jax.random.key(1), (B, L, K))
    v = vector_field_from_logits(logits, xt, jnp.full((B,), 0.5))
    assert jnp.allclose(v.sum(-1), 0.0, atol=1e-5), f"Velocity is not tangent: {v.sum(-1)}"