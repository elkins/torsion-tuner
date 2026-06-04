import os
from typing import Any

import biotite.structure as stripe
import numpy as np

from torsiontuner.export import save_for_psvs


def test_save_for_psvs(tmp_path: Any) -> None:
    # Create a dummy structure
    atom1 = stripe.Atom(
        coord=[0, 0, 0], chain_id="", res_id=1, res_name="ALA", atom_name="N", element="N"
    )
    atom2 = stripe.Atom(
        coord=[1, 1, 1], chain_id="", res_id=1, res_name="ALA", atom_name="CA", element="C"
    )
    # Add a non-amino acid atom to test filtering
    atom3 = stripe.Atom(
        coord=[2, 2, 2],
        chain_id="",
        res_id=2,
        res_name="HOH",
        atom_name="O",
        element="O",
        hetero=True,
    )

    struct = stripe.array([atom1, atom2, atom3])

    file_path = os.path.join(tmp_path, "test_psvs.pdb")
    save_for_psvs(struct, file_path)

    assert os.path.exists(file_path)

    # Reload and verify
    import biotite.structure.io.pdb as pdb

    pdb_file = pdb.PDBFile.read(file_path)
    reloaded_struct = pdb_file.get_structure()

    if isinstance(reloaded_struct, stripe.AtomArrayStack):
        reloaded_struct = reloaded_struct[0]

    # Should only have 2 atoms (ALA), and chain ID should be "A"
    assert len(reloaded_struct) == 2
    assert np.all(reloaded_struct.chain_id == "A")
    assert "HOH" not in reloaded_struct.res_name
