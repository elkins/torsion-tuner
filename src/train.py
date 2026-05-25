import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx
from src.data import load_pdb, get_graph_features
from src.model import FineTunerGNN
from src.kinematics import rebuild_backbone
from diff_biophys.saxs import debye_saxs

def train():
    # 1. Load data
    data = load_pdb("test_helix.pdb")
    node_features, adj = get_graph_features(data)
    
    # Target: let's say we want to recover the original coordinates (mocking SAXS)
    # Generate synthetic SAXS target from "true" data
    q_values = jnp.linspace(0.01, 0.5, 50)
    # Simple form factors (all atoms are Carbon for now)
    form_factors = jnp.ones((len(data['coords']), len(q_values)))
    target_saxs = debye_saxs(data['coords'], q_values, form_factors)
    
    # 2. Initialize Model
    key = jr.PRNGKey(42)
    model_key, train_key = jr.split(key)
    model = FineTunerGNN(
        node_dim=20, # Residue types
        hidden_dim=64,
        out_dim=2, # delta_phi, delta_psi
        n_layers=3,
        key=model_key
    )
    
    # 3. Optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=5e-4)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # 4. Loss Function
    def loss_fn(model, node_features, adj, data, target_saxs, q_values, form_factors):
        # Predict deltas
        deltas = model(node_features, adj)
        
        # Reconstruct
        refined_coords = rebuild_backbone(
            data['init_coords'],
            data['lengths'],
            data['angles'],
            data['dihedrals'],
            deltas
        )
        
        # Calculate SAXS
        pred_saxs = debye_saxs(refined_coords, q_values, form_factors)
        
        # Normalize intensities for loss
        scale = jnp.max(target_saxs)
        saxs_loss = jnp.mean(((pred_saxs - target_saxs) / scale)**2)
        
        # Regularization: keep deltas small
        reg_loss = jnp.mean(deltas**2)
        
        total_loss = saxs_loss + 0.01 * reg_loss
        return total_loss

    @eqx.filter_jit
    def make_step(model, opt_state, node_features, adj, data, target_saxs, q_values, form_factors):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(
            model, node_features, adj, data, target_saxs, q_values, form_factors
        )
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    # 5. Training Loop
    print("Starting training...")
    for step in range(101):
        model, opt_state, loss = make_step(
            model, opt_state, node_features, adj, data, target_saxs, q_values, form_factors
        )
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss:.6f}")
            
    print("Training complete.")
    
    # Save final structure
    final_deltas = model(node_features, adj)
    final_coords = rebuild_backbone(
        data['init_coords'],
        data['lengths'],
        data['angles'],
        data['dihedrals'],
        final_deltas
    )
    
    import biotite.structure as stripe
    import biotite.structure.io.pdb as pdb
    import numpy as np
    
    # Create Biotite structure from final_coords
    atoms = []
    res_indices = data['res_indices']
    for i in range(len(res_indices)):
        for j, name in enumerate(['N', 'CA', 'C']):
            idx = i * 3 + j
            atom = stripe.Atom(
                coord=np.array(final_coords[idx]),
                chain_id='A',
                res_id=i+1,
                res_name='ALA',
                atom_name=name,
                element='C' if name != 'N' else 'N'
            )
            atoms.append(atom)
            
    final_struct = stripe.array(atoms)
    pdb_file = pdb.PDBFile()
    pdb_file.set_structure(final_struct)
    pdb_file.write("refined_helix.pdb")
    print("Saved refined_helix.pdb")
    
    return model

if __name__ == "__main__":
    train()
