from torsiontuner.psvs_parser import parse_psvs_summary


def test_parse_psvs_summary_full() -> None:
    """Test parsing a complete PSVS summary."""
    text = """
    PSVS 1.5 Analysis
    Verify3D (expected > -0.1): 0.25
    PROCHECK G-factor (phi/psi) (expected > -0.5): -0.20
    PROCHECK G-factor (all-atom) (expected > -0.5): -0.40
    MolProbity clashscore (expected < 10): 5.5
    """
    results = parse_psvs_summary(text)
    assert results["verify3d"] == 0.25
    assert results["procheck_phi_psi"] == -0.20
    assert results["procheck_all"] == -0.40
    assert results["clashscore"] == 5.5


def test_parse_psvs_summary_partial() -> None:
    """Test parsing a PSVS summary with missing fields."""
    text = "Verify3D (expected > -0.1): 0.1"
    results = parse_psvs_summary(text)
    assert results["verify3d"] == 0.1
    assert "clashscore" not in results
    assert "procheck_phi_psi" not in results


def test_parse_psvs_summary_empty() -> None:
    """Test parsing an empty or irrelevant string."""
    assert parse_psvs_summary("") == {}
    assert parse_psvs_summary("No metrics here") == {}


def test_parse_psvs_summary_negative_values() -> None:
    """Test parsing negative metric values."""
    text = "PROCHECK G-factor (phi/psi) (expected > -0.5): -1.2"
    results = parse_psvs_summary(text)
    assert results["procheck_phi_psi"] == -1.2
