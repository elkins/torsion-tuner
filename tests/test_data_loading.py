from typing import Any

from torsiontuner.data import get_graph_features, load_pdb


def test_load_pdb() -> None:
    data = load_pdb("test_helix.pdb")
    assert "coords" in data
    assert "res_indices" in data
    assert "elements" in data
    assert len(data["res_indices"]) == 20
    assert data["coords"].shape == (60, 3)  # 20 residues * 3 atoms
    assert len(data["elements"]) == 60


def test_load_pdb_stack(tmp_path: Any) -> None:
    import biotite.structure as stripe
    import biotite.structure.io.pdb as pdb

    # Create a dummy structure
    atom1 = stripe.Atom(coord=[0, 0, 0], res_id=1, res_name="ALA", atom_name="N", element="N")
    atom2 = stripe.Atom(coord=[1, 1, 1], res_id=1, res_name="ALA", atom_name="CA", element="C")
    atom3 = stripe.Atom(coord=[2, 2, 2], res_id=1, res_name="ALA", atom_name="C", element="C")
    struct = stripe.array([atom1, atom2, atom3])

    # Create a stack with 2 models
    stack = stripe.stack([struct, struct])

    file_path = str(tmp_path / "stack.pdb")
    pdb_file = pdb.PDBFile()
    pdb_file.set_structure(stack)
    pdb_file.write(file_path)

    data = load_pdb(file_path)
    assert data["coords"].shape == (3, 3)


def test_graph_features() -> None:
    data = load_pdb("test_helix.pdb")
    node_features, adj, edge_features = get_graph_features(data)

    n_residues = 20
    assert node_features.shape == (n_residues, 20)
    assert adj.shape == (n_residues, n_residues)
    assert edge_features.shape == (n_residues, n_residues, 1)

    # Check spatial adjacency
    # In a helix, residue i and i+4 should be close (approx 5.4A)
    # 10A cutoff should definitely include i and i+1, i+2, i+3, i+4
    assert adj[0, 1] == 1.0
    assert adj[0, 4] == 1.0


if __name__ == "__main__":
    test_load_pdb()
    test_graph_features()
    print("Data loading tests passed!")
