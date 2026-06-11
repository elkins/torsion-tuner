import jax.numpy as jnp
import numpy as np
from diff_biophys.geometry import chain_nerf, compute_dihedrals
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


def test_kinematics_roundtrip() -> None:
    """
    Property-based test: Reconstructing from geometry and then
    computing geometry should yield original values (for dihedrals).
    """
    # Fixed initial coords for a stable NeRF start
    init_coords = jnp.array([[0.0, 0.0, 0.0], [1.46, 0.0, 0.0], [2.0, 1.4, 0.0]])

    n_residues = 5
    n_atoms_to_place = 3 * n_residues - 3

    # Generate random but physical geometry
    lengths = jnp.full((n_atoms_to_place,), 1.5)
    angles = jnp.full((n_atoms_to_place,), jnp.deg2rad(110.0))

    @given(dihedrals=arrays(np.float32, (n_atoms_to_place,), elements=st.floats(-np.pi, np.pi)))
    def check_dihedrals(dihedrals: np.ndarray) -> None:
        coords = chain_nerf(init_coords, lengths, angles, jnp.array(dihedrals))
        # compute_dihedrals returns dihedrals starting from the 4th atom
        calc_dihedrals = compute_dihedrals(coords)

        # Wrap dihedrals to [-pi, pi] to handle periodicity
        diff = np.array(calc_dihedrals) - dihedrals
        diff = (diff + np.pi) % (2 * np.pi) - np.pi

        np.testing.assert_allclose(diff, 0.0, atol=1e-4)

    check_dihedrals()


if __name__ == "__main__":
    test_kinematics_roundtrip()
