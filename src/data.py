import jax
import jax.numpy as jnp
import numpy as np
import biotite.structure as stripe
import biotite.structure.io as stripeio
from diff_biophys.geometry import compute_dihedrals, compute_bond_lengths, compute_bond_angles

RESIDUE_TYPES = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]
RESIDUE_MAP = {res: i for i, res in enumerate(RESIDUE_TYPES)}

def load_pdb(file_path):
    """Load a PDB file and extract backbone information."""
    struct = stripeio.load_structure(file_path)
    if isinstance(struct, stripe.AtomArrayStack):
        struct = struct[0]
        
    # Filter for backbone atoms (N, CA, C)
    mask = np.isin(struct.atom_name, ['N', 'CA', 'C'])
    backbone = struct[mask]
    
    res_names = []
    coords = []
    
    # Iterate through residues
    res_ids = np.unique(backbone.res_id)
    for rid in res_ids:
        res_atoms = backbone[backbone.res_id == rid]
        # Expect N, CA, C
        n = res_atoms[res_atoms.atom_name == 'N']
        ca = res_atoms[res_atoms.atom_name == 'CA']
        c = res_atoms[res_atoms.atom_name == 'C']
        
        if len(n) == 1 and len(ca) == 1 and len(c) == 1:
            res_names.append(res_atoms.res_name[0])
            coords.extend([n.coord[0], ca.coord[0], c.coord[0]])
            
    coords = jnp.array(coords)
    res_indices = jnp.array([RESIDUE_MAP.get(name, 0) for name in res_names])
    
    # Initial geometry for NeRF
    init_coords = coords[:3]
    lengths = compute_bond_lengths(coords)
    angles = compute_bond_angles(coords)
    dihedrals = compute_dihedrals(coords)
    
    return {
        'coords': coords,
        'res_indices': res_indices,
        'init_coords': init_coords,
        'lengths': lengths,
        'angles': angles,
        'dihedrals': dihedrals
    }

def get_graph_features(data):
    """Convert loaded data into graph features."""
    res_indices = data['res_indices']
    coords = data['coords']
    n_residues = len(res_indices)
    
    # Node features: residue type
    node_features = jax.nn.one_hot(res_indices, 20)
    
    # Sequence adjacency
    seq_adj = jnp.eye(n_residues, k=1) + jnp.eye(n_residues, k=-1)
    
    # Spatial adjacency (CA-CA)
    ca_coords = coords[1::3]
    dist_matrix = jnp.sqrt(jnp.sum((ca_coords[:, None, :] - ca_coords[None, :, :])**2, axis=-1))
    spatial_adj = (dist_matrix < 10.0).astype(jnp.float32)
    
    adj = (seq_adj + spatial_adj > 0).astype(jnp.float32)
    
    return node_features, adj

if __name__ == "__main__":
    import jax
    # Quick test if possible
    print("Data module ready.")
