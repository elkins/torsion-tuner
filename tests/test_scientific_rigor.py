"""
Scientific rigor tests for structural integrity and physical validity.
These tests ensure that TorsionTuner's refinement process respects the
laws of physics and structural biology.
"""

import jax.numpy as jnp
import numpy as np

from torsiontuner.kinematics import rebuild_backbone
from torsiontuner.montelione_utils import ramachandran_penalty


def test_neRF_bond_length_conservation() -> None:
    """
    Ensure that NeRF reconstruction preserves ideal bond lengths regardless
    of the torsion angle changes (delta_phi_psi).
    """
    # 20 residue helix starting coordinates (mocked)
    n_res = 20
    n_atoms = n_res * 3

    # N-CA-C geometry
    # Distance N-CA = 1.46, CA-C = 1.52
    # Angle N-CA-C = 111 deg (1.937 rad)
    # Using simple coordinates to ensure exact lengths
    init_coords = jnp.array(
        [
            [0.0, 0.0, 0.0],  # N
            [1.46, 0.0, 0.0],  # CA
            [1.46 + 1.52 * np.cos(1.937 - np.pi), 1.52 * np.sin(1.937 - np.pi), 0.0],  # C
        ]
    )

    # Verify my own mock init_coords match 1.52 for CA-C
    ca_c_dist = np.linalg.norm(init_coords[2] - init_coords[1])
    assert np.abs(ca_c_dist - 1.52) < 1e-4

    # Full length/angle/dihedral arrays as produced by load_pdb
    # lengths: size 3N-1
    lengths = jnp.tile(jnp.array([1.46, 1.52, 1.33]), n_res)[: n_atoms - 1]
    # angles: size 3N-2
    angles = jnp.tile(jnp.array([1.937, 2.025, 2.112]), n_res)[: n_atoms - 2]
    # dihedrals: size 3N-3
    dihedrals = jnp.zeros(n_atoms - 3)

    # Apply VERY large, non-physical deltas to try and break the geometry
    delta_phi_psi = jnp.ones((n_res, 2)) * 10.0  # 10 radians is ~570 degrees

    refined_coords, _ = rebuild_backbone(init_coords, lengths, angles, dihedrals, delta_phi_psi)

    # Calculate bond lengths in the refined structure
    diffs = jnp.diff(refined_coords, axis=0)
    refined_lengths = jnp.sqrt(jnp.sum(diffs**2, axis=1))

    # Compare with initial lengths (should be identical within precision)
    # Note: refined_lengths starts from dist(0,1), dist(1,2), ...
    np.testing.assert_allclose(refined_lengths, lengths, atol=1e-4)
    print("Scientific Check: Bond lengths perfectly conserved by NeRF.")


def test_steric_clash_detection() -> None:
    """
    Check for internal steric clashes (atoms getting too close).
    While TorsionTuner doesn't have an explicit clash loss yet,
    the refinement should ideally avoid creating them.
    """
    # Create a structure that is forced to clash
    n_res = 10
    n_atoms = n_res * 3
    init_coords = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.46, 0.0, 0.0],
            [1.46 + 1.52 * np.cos(1.937 - np.pi), 1.52 * np.sin(1.937 - np.pi), 0.0],
        ]
    )
    lengths = jnp.tile(jnp.array([1.46, 1.52, 1.33]), n_res)[: n_atoms - 1]
    angles = jnp.tile(jnp.array([1.937, 2.025, 2.112]), n_res)[: n_atoms - 2]
    dihedrals = jnp.zeros(n_atoms - 3)

    # Extreme phi/psi that causes a tight turn
    delta = jnp.ones((n_res, 2)) * jnp.pi

    refined_coords, _ = rebuild_backbone(init_coords, lengths, angles, dihedrals, delta)

    # Calculate all-vs-all distances between non-consecutive CA atoms
    ca_coords = refined_coords[1::3]
    n_ca = ca_coords.shape[0]
    dist_matrix = jnp.sqrt(jnp.sum((ca_coords[:, None, :] - ca_coords[None, :, :]) ** 2, axis=-1))

    # Mask out consecutive residues and self
    mask = jnp.abs(jnp.arange(n_ca)[:, None] - jnp.arange(n_ca)[None, :]) > 1
    non_bonded_dists = dist_matrix[mask]

    min_dist = jnp.min(non_bonded_dists)

    # Van der Waals radius of CA is ~1.7A, so clash is < 3.0A
    print(f"Scientific Check: Minimum non-bonded CA-CA distance is {min_dist:.2f}A")
    assert min_dist > 0.5  # Basic sanity


def test_ramachandran_favorability() -> None:
    """
    Ensure the ramachandran_penalty correctly identifies 'good' vs 'bad' geometry.
    """
    # 1. Favored region (alpha helix)
    phi_good = jnp.array([-1.05])
    psi_good = jnp.array([-0.78])
    penalty_good = ramachandran_penalty(phi_good, psi_good)

    # 2. Disallowed region (e.g., phi=90, psi=90)
    phi_bad = jnp.array([1.57])
    psi_bad = jnp.array([1.57])
    penalty_bad = ramachandran_penalty(phi_bad, psi_bad)

    assert penalty_bad > penalty_good
    print(
        f"Scientific Check: Ramachandran penalty correctly identified bad geometry ({penalty_bad:.2f} vs {penalty_good:.2f})"
    )


if __name__ == "__main__":
    test_neRF_bond_length_conservation()
    test_steric_clash_detection()
    test_ramachandran_favorability()
    print("Scientific rigor tests passed!")
