import os
from collections.abc import Iterator

import pytest
from synth_pdb.generator import PeptideGenerator


@pytest.fixture(scope="session", autouse=True)
def ensure_test_helix() -> Iterator[None]:
    """Ensure test_helix.pdb exists for the duration of the test session."""
    if not os.path.exists("test_helix.pdb"):
        generator = PeptideGenerator(sequence="A" * 20, conformation="alpha")
        result = generator.generate()
        result.save("test_helix.pdb")
    yield
    # We keep it for inspection if needed, or we could delete it here
