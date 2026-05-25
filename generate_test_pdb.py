import jax.numpy as jnp
import numpy as np
import biotite.structure as stripe
import biotite.structure.io.pdb as pdb
from diff_biophys.geometry import chain_nerf

def generate_real_helix(n_residues=20):
    # Setup parameters for atoms 4 to N
    n_atoms = n_residues * 3
    lengths = jnp.tile(jnp.array([1.46, 1.52, 1.33]), n_residues)[:n_atoms-3]
    angles = jnp.tile(jnp.radians(jnp.array([111.0, 116.0, 121.0])), n_residues)[:n_atoms-3]
    
    # alpha helix: phi=-57, psi=-47, omega=180
    # Mapping:
    # atoms: N1, CA1, C1, N2, CA2, C2, ...
    # dihedrals:
    # 0: psi1 (-47)
    # 1: omega1 (180)
    # 2: phi2 (-57)
    # ...
    dihedrals = jnp.tile(jnp.radians(jnp.array([-47.0, 180.0, -57.0])), n_residues)[:n_atoms-3]
    
    init_coords = jnp.array([
        [0.0, 0.0, 0.0],
        [1.46, 0.0, 0.0],
        [1.46 + 1.52 * jnp.cos(jnp.radians(180-111)), 1.52 * jnp.sin(jnp.radians(180-111)), 0.0]
    ])
    
    coords = chain_nerf(init_coords, lengths, angles, dihedrals)
    
    # Create Biotite structure
    atoms = []
    for i in range(n_residues):
        for j, name in enumerate(['N', 'CA', 'C']):
            idx = i * 3 + j
            atom = stripe.Atom(
                coord=np.array(coords[idx]),
                chain_id='A',
                res_id=i+1,
                res_name='ALA',
                atom_name=name,
                element='C' if name != 'N' else 'N'
            )
            atoms.append(atom)
            
    struct = stripe.array(atoms)
    return struct

if __name__ == "__main__":
    helix = generate_real_helix()
    pdb_file = pdb.PDBFile()
    pdb_file.set_structure(helix)
    pdb_file.write("test_helix.pdb")
    print("Generated real test_helix.pdb")
