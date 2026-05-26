import os

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax
import pandas as pd
from diff_biophys.nmr.chemical_shifts import predict_ca_shifts

from torsiontuner.data import get_graph_features, load_pdb
from torsiontuner.kinematics import rebuild_backbone
from torsiontuner.model import FineTunerGNN
from torsiontuner.montelione_utils import ramachandran_penalty, get_residue_rc_shifts


def test_2rn7_benchmark():
    """
    Scientific Benchmark: 2RN7 (NESG SfR125).
    Verify that refinement reduces CSRMSD against experimental BMRB 11017 data.
    This target is a 108-residue TnpE protein (IS629 orfA) from Shigella flexneri.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data", "2rn7")
    shift_data = pd.read_csv(os.path.join(data_dir, "shifts.csv"))
    pdb_path = os.path.join(data_dir, "starting_model.pdb")

    # 1. Load the actual AlphaFold model
    data = load_pdb(pdb_path)
    node_features, adj, edge_features = get_graph_features(data)
    res_indices = data["res_indices"]

    # 2. Align Experimental Data
    # BMRB shifts are provided for specific residue numbers.
    # We'll create a mask to compare only where we have experimental data.
    target_res_ids = shift_data["residue"].values
    target_shifts = jnp.array(shift_data["shift"].values)

    rc_shifts_full = get_residue_rc_shifts(res_indices)

    # Map target residue IDs to our 0-indexed arrays
    # 2RN7 PDB usually starts at res 1.
    # res_indices starts from residue 1 (index 0).
    # So residue N is index N-1.

    def get_benchmark_cs_rmsd(phi, psi):
        # Full predictions
        # res_indices[1:] because first residue doesn't have phi/psi
        # Actually, our model predicts phi/psi for each node.
        # But NeRF usually starts from residue 2 for phi/psi.
        # Alignment must be surgical.

        # In torsiontuner, pred_phi[i] is for residue i+1 (1-indexed)
        # So residue N is pred_phi[N-1]

        # Get predictions for all
        rc_subset = rc_shifts_full[1:]
        pred_shifts_all = predict_ca_shifts(phi, psi, rc_subset)

        # pred_shifts_all[0] corresponds to residue 2
        # So pred_shifts_all[N-2] corresponds to residue N

        # Map target_res_ids (5, 6, 7...) to indices in pred_shifts_all
        # res 5 -> pred_shifts_all[3]
        mapped_indices = target_res_ids - 2

        # Filter for valid indices (some targets might be outside our prediction range)
        valid_mask = (mapped_indices >= 0) & (mapped_indices < len(pred_shifts_all))
        final_mapped = mapped_indices[valid_mask]
        final_targets = target_shifts[valid_mask]

        matched_preds = pred_shifts_all[final_mapped]
        return jnp.sqrt(jnp.mean((matched_preds - final_targets) ** 2))

    # Initial State
    phi_init = data["dihedrals"][2::3]
    psi_init = data["dihedrals"][0::3]
    initial_cs_rmsd = get_benchmark_cs_rmsd(phi_init, psi_init)
    print(f"Initial CSRMSD: {initial_cs_rmsd:.4f} ppm")

    # 3. Initialize Model and Optimizer
    key = jr.PRNGKey(42)
    model = FineTunerGNN(node_dim=20, hidden_dim=64, out_dim=2, n_layers=3, key=key)
    optimizer = optax.adamw(learning_rate=1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # 4. Refinement Loop
    def loss_fn(model):
        deltas = model(node_features, adj, edge_features)
        _, updated_dihedrals = rebuild_backbone(
            data["init_coords"],
            data["lengths"],
            data["angles"],
            data["dihedrals"],
            deltas,
        )
        pred_phi = updated_dihedrals[2::3]
        pred_psi = updated_dihedrals[0::3]

        # Calculate CSRMSD on the subset
        rc_subset = rc_shifts_full[1:]
        pred_shifts_all = predict_ca_shifts(pred_phi, pred_psi, rc_subset)

        mapped_indices = target_res_ids - 2
        valid_mask = (mapped_indices >= 0) & (mapped_indices < len(pred_shifts_all))
        final_mapped = mapped_indices[valid_mask]
        final_targets = target_shifts[valid_mask]

        matched_preds = pred_shifts_all[final_mapped]
        cs_loss = jnp.mean((matched_preds - final_targets) ** 2)

        ansurr_loss = ramachandran_penalty(pred_phi, pred_psi)
        reg_loss = jnp.mean(deltas**2)

        return 1.0 * cs_loss + 0.1 * ansurr_loss + 0.01 * reg_loss

    @eqx.filter_jit
    def make_step(model, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    print("Running refinement...")
    for step in range(51):
        model, opt_state, loss = make_step(model, opt_state)
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")

    # 5. Final Validation
    final_deltas = model(node_features, adj, edge_features)
    _, final_dihedrals = rebuild_backbone(
        data["init_coords"],
        data["lengths"],
        data["angles"],
        data["dihedrals"],
        final_deltas,
    )
    final_phi = final_dihedrals[2::3]
    final_psi = final_dihedrals[0::3]
    final_cs_rmsd = get_benchmark_cs_rmsd(final_phi, final_psi)

    print(f"Final CSRMSD: {final_cs_rmsd:.4f} ppm")

    # Success Criterion
    assert final_cs_rmsd < initial_cs_rmsd, "Refinement failed to reduce CSRMSD"
    print("Scientific Benchmark 2RN7: PASSED")


if __name__ == "__main__":
    test_2rn7_benchmark()
