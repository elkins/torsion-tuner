import jax.numpy as jnp

from torsiontuner.data import load_pdb
from torsiontuner.kinematics import rebuild_backbone


def test_kinematics_reconstruction():
    # 1. Load the test helix
    data = load_pdb("test_helix.pdb")
    coords = data["coords"]
    n_residues = len(data["res_indices"])

    # 2. No deltas
    deltas = jnp.zeros((n_residues, 2))

    # 3. Reconstruct
    refined_coords, _ = rebuild_backbone(
        data["init_coords"], data["lengths"], data["angles"], data["dihedrals"], deltas
    )

    # 4. Check if reconstructed coords match original
    # Note: NeRF might have very slight numerical differences
    max_diff = jnp.max(jnp.abs(refined_coords - coords))
    print(f"Max difference between original and reconstructed: {max_diff:.6e}")

    assert max_diff < 1e-4, f"Reconstruction failed, max diff: {max_diff}"
    print("Kinematics reconstruction test passed!")


if __name__ == "__main__":
    test_kinematics_reconstruction()
