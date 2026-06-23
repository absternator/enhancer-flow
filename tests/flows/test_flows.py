from enhancer_flow.flows import sample_dirichlet_path, denoising_ce_loss
import jax
import jax.numpy as jnp
from enhancer_flow.data import VOCAB_SIZE
from enhancer_flow.flows.paths import vector_field_from_logits

def test_dirichlet_path_on_simplex():
    key = jax.random.key(0)
    x1 = jnp.array([[0, 1, 2, 3, 0, 1]])  # (1, 6)
    t = jnp.array([0.5])
    xt = sample_dirichlet_path(key, x1, t)
    assert xt.shape == (1, 6, VOCAB_SIZE)
    assert jnp.allclose(xt.sum(-1), 1.0, atol=1e-5)
    assert jnp.all(xt > 0) and jnp.all(xt < 1)


def test_path_concentrates_with_time():
    """At larger t the path should put more mass on the true class on average."""
    key = jax.random.key(1)
    x1 = jnp.zeros((2048, 1), dtype=jnp.int32)  # all class 0
    lo = sample_dirichlet_path(key, x1, jnp.full((2048,), 0.1))
    hi = sample_dirichlet_path(key, x1, jnp.full((2048,), 5.0), alpha_scale=4.0)
    assert hi[..., 0].mean() > lo[..., 0].mean()


def test_ce_loss_beats_uniform_when_confident():
    x1 = jnp.array([[0, 1, 2, 3]])
    confident = jax.nn.one_hot(x1, VOCAB_SIZE) * 10.0  # peaked at truth
    uniform = jnp.zeros((1, 4, VOCAB_SIZE))
    assert float(denoising_ce_loss(confident, x1)) < float(
        denoising_ce_loss(uniform, x1)
    )

def test_velocity_is_tangent():
    """Velocity rows must sum to ~0 so integration stays on the simplex."""
    key = jax.random.key(2)
    x1 = jnp.array([[0, 1, 2, 3]])
    t = jnp.array([0.5])
    xt = sample_dirichlet_path(key, x1, t)
    logits = jnp.zeros((1, 4, VOCAB_SIZE))
    v = vector_field_from_logits(logits, xt, t)
    assert jnp.allclose(v.sum(-1), 0.0, atol=1e-5)