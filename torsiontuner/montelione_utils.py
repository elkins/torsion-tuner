import jax
import jax.numpy as jnp
from diff_biophys.nmr.chemical_shifts import RANDOM_COIL_CA, predict_ca_shifts

from torsiontuner.data import RESIDUE_TYPES

# Pre-compute a JAX array for random coil shifts to allow JIT-compatible indexing
_RC_ARRAY = jnp.array([RANDOM_COIL_CA[res] for res in RESIDUE_TYPES])


def get_residue_rc_shifts(res_indices):
    """Map residue indices to their random coil CA shifts."""
    # Use JAX indexing instead of list comprehension
    return _RC_ARRAY[res_indices]


def montelione_loss(phi, psi, target_shifts, res_indices):
    """
    A loss function inspired by the Montelione group's approach of
    validating structures against experimental chemical shifts.
    """
    rc_shifts = get_residue_rc_shifts(res_indices)
    pred_shifts = predict_ca_shifts(phi, psi, rc_shifts)

    # MSE between predicted and experimental CA shifts
    return jnp.mean((pred_shifts - target_shifts) ** 2)


def ramachandran_penalty(phi, psi):
    """
    A soft Ramachandran-region penalty used as a backbone geometry regularizer.

    Penalizes phi/psi values that fall far from the three canonical
    favored regions of the Ramachandran plot (alpha-helix, beta-strand,
    and left-handed alpha-helix), using a smooth log-sum-exp potential.

    This is analogous to the Ramachandran term used in programs such as
    CNS and Rosetta (e.g., Engh & Huber torsion potentials), and is
    similar in spirit to the G-factor computed by PROCHECK.

    Note: This is NOT related to the ANSURR validation method (Fowler,
    Sljoka & Williamson, 2020, Nature Commun. 11:6321). ANSURR measures
    agreement between experimental NMR backbone flexibility (RCI) and
    structural rigidity computed by graph-theoretic rigidity theory
    (FIRST algorithm); it cannot be computed from torsion angles alone.

    Args:
        phi: (N,) backbone phi angles in radians.
        psi: (N,) backbone psi angles in radians.

    Returns:
        Scalar penalty (lower = more geometrically regular backbone).
    """
    # Alpha-helix region: phi ~ -60 deg (-1.05 rad), psi ~ -45 deg (-0.78 rad)
    alpha_dist = jnp.sqrt((phi + 1.05) ** 2 + (psi + 0.78) ** 2)
    # Beta-strand region: phi ~ -120 deg (-2.09 rad), psi ~ +135 deg (+2.35 rad)
    beta_dist = jnp.sqrt((phi + 2.09) ** 2 + (psi - 2.35) ** 2)
    # Left-handed alpha-helix: phi ~ +60 deg (+1.05 rad), psi ~ +45 deg (+0.78 rad)
    l_alpha_dist = jnp.sqrt((phi - 1.05) ** 2 + (psi - 0.78) ** 2)

    # Soft minimum distance to any favored region via log-sum-exp
    sigma = 0.5
    dist_sq = jnp.stack([alpha_dist, beta_dist, l_alpha_dist], axis=-1) ** 2
    penalty = -sigma * jax.nn.logsumexp(-dist_sq / sigma, axis=-1)

    return jnp.mean(penalty)
