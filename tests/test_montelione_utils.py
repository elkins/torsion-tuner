import jax.numpy as jnp

from torsiontuner.montelione_utils import (
    get_residue_rc_shifts,
    montelione_loss,
    ramachandran_penalty,
)


def test_get_residue_rc_shifts():
    res_indices = jnp.array([0, 1, 2])  # ALA, ARG, ASN
    shifts = get_residue_rc_shifts(res_indices)
    assert shifts.shape == (3,)
    assert not jnp.any(jnp.isnan(shifts))


def test_montelione_loss():
    n_res = 10
    phi = jnp.zeros(n_res)
    psi = jnp.zeros(n_res)
    target_shifts = jnp.zeros(n_res)
    res_indices = jnp.zeros(n_res, dtype=jnp.int32)

    loss = montelione_loss(phi, psi, target_shifts, res_indices)
    assert loss >= 0
    assert not jnp.isnan(loss)


def test_ramachandran_penalty():
    n_res = 5
    phi = jnp.array([-1.05] * n_res)  # Near Alpha region
    psi = jnp.array([-0.78] * n_res)

    loss_favorable = ramachandran_penalty(phi, psi)

    phi_bad = jnp.array([1.0] * n_res)  # Typically unfavorable
    psi_bad = jnp.array([1.0] * n_res)
    loss_unfavorable = ramachandran_penalty(phi_bad, psi_bad)

    # Lower is better in this proxy implementation (penalty)
    assert loss_favorable < loss_unfavorable
