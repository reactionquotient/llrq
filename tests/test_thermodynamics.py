"""Tests for thermodynamic utilities."""

import numpy as np
import pytest
import warnings

from llrq.utils.thermodynamics import (
    delta_g_to_keq,
    keq_to_delta_g,
    compute_reaction_delta_g,
    metabolite_delta_g_to_reaction_keq,
    validate_thermodynamic_consistency,
    R_JOULE_MOL_K,
)


class TestDeltaGToKeq:
    """Test ΔG° to Keq conversion."""

    def test_basic_conversion(self):
        """Test basic ΔG° to Keq conversion."""
        # ΔG° = 0 should give Keq = 1
        assert delta_g_to_keq(0.0) == pytest.approx(1.0)

        # Negative ΔG° should give Keq > 1
        keq = delta_g_to_keq(-10.0)  # kJ/mol
        assert keq > 1.0
        assert keq == pytest.approx(55.34, rel=1e-2)

        # Positive ΔG° should give Keq < 1
        keq = delta_g_to_keq(10.0)  # kJ/mol
        assert keq < 1.0
        assert keq == pytest.approx(0.0181, rel=1e-2)

    def test_temperature_dependence(self):
        """Test temperature dependence of conversion."""
        delta_g = -10.0  # kJ/mol

        # Higher temperature should give smaller Keq for negative ΔG°
        keq_298 = delta_g_to_keq(delta_g, T=298.15)
        keq_373 = delta_g_to_keq(delta_g, T=373.15)  # 100°C

        assert keq_373 < keq_298

    def test_units(self):
        """Test different units."""
        delta_g_kj = -10.0  # kJ/mol
        delta_g_kcal = delta_g_kj / 4.184  # kcal/mol
        delta_g_j = delta_g_kj * 1000  # J/mol

        keq_kj = delta_g_to_keq(delta_g_kj, units="kJ/mol")
        keq_kcal = delta_g_to_keq(delta_g_kcal, units="kcal/mol")
        keq_j = delta_g_to_keq(delta_g_j, units="J/mol")

        assert keq_kj == pytest.approx(keq_kcal, rel=1e-6)
        assert keq_kj == pytest.approx(keq_j, rel=1e-6)

    def test_invalid_units(self):
        """Test invalid units raise error."""
        with pytest.raises(ValueError, match="Unsupported units"):
            delta_g_to_keq(-10.0, units="eV")

    def test_array_input(self):
        """Test array input."""
        delta_g_array = np.array([-10.0, 0.0, 10.0])
        keq_array = delta_g_to_keq(delta_g_array)

        assert len(keq_array) == 3
        assert keq_array[0] > 1.0  # Favorable
        assert keq_array[1] == pytest.approx(1.0)  # Equilibrium
        assert keq_array[2] < 1.0  # Unfavorable


class TestKeqToDeltaG:
    """Test Keq to ΔG° conversion."""

    def test_basic_conversion(self):
        """Test basic Keq to ΔG° conversion."""
        # Keq = 1 should give ΔG° = 0
        assert keq_to_delta_g(1.0) == pytest.approx(0.0)

        # Keq > 1 should give negative ΔG°
        delta_g = keq_to_delta_g(55.34)
        assert delta_g < 0.0
        assert delta_g == pytest.approx(-10.0, rel=1e-2)

        # Keq < 1 should give positive ΔG°
        delta_g = keq_to_delta_g(0.0181)
        assert delta_g > 0.0
        assert delta_g == pytest.approx(10.0, rel=1e-2)

    def test_roundtrip_conversion(self):
        """Test roundtrip conversion maintains values."""
        original_delta_g = np.array([-20.0, -5.0, 0.0, 5.0, 20.0])

        # Convert to Keq and back
        keq = delta_g_to_keq(original_delta_g)
        recovered_delta_g = keq_to_delta_g(keq)

        np.testing.assert_allclose(original_delta_g, recovered_delta_g, rtol=1e-10)

    def test_invalid_keq(self):
        """Test invalid Keq values raise error."""
        with pytest.raises(ValueError, match="Equilibrium constants must be positive"):
            keq_to_delta_g(-1.0)

        with pytest.raises(ValueError, match="Equilibrium constants must be positive"):
            keq_to_delta_g(0.0)


class TestComputeReactionDeltaG:
    """Test reaction ΔG° computation from metabolite formation energies."""

    def test_simple_reaction(self):
        """Test simple A -> B reaction."""
        # A -> B where ΔG°_A = -100 kJ/mol, ΔG°_B = -120 kJ/mol
        # Reaction ΔG° = ΔG°_B - ΔG°_A = -120 - (-100) = -20 kJ/mol

        metabolite_dg = {"A": -100.0, "B": -120.0}
        stoichiometry = {"A": -1, "B": 1}  # A consumed, B produced

        rxn_dg, missing = compute_reaction_delta_g(metabolite_dg, stoichiometry)

        assert rxn_dg == pytest.approx(-20.0)
        assert len(missing) == 0

    def test_complex_reaction(self):
        """Test complex reaction: 2A + B -> C + 3D."""
        metabolite_dg = {
            "A": -100.0,  # 2 * (-100) = -200
            "B": -50.0,  # 1 * (-50) = -50
            "C": -200.0,  # 1 * (-200) = -200
            "D": -80.0,  # 3 * (-80) = -240
        }
        stoichiometry = {"A": -2, "B": -1, "C": 1, "D": 3}

        # Products: -200 + 3*(-80) = -200 - 240 = -440
        # Reactants: 2*(-100) + 1*(-50) = -200 - 50 = -250
        # ΔG°_rxn = -440 - (-250) = -190 kJ/mol

        rxn_dg, missing = compute_reaction_delta_g(metabolite_dg, stoichiometry)

        assert rxn_dg == pytest.approx(-190.0)
        assert len(missing) == 0

    def test_missing_metabolite(self):
        """Test handling of missing metabolite ΔG° data."""
        metabolite_dg = {"A": -100.0}  # B is missing
        stoichiometry = {"A": -1, "B": 1}

        rxn_dg, missing = compute_reaction_delta_g(metabolite_dg, stoichiometry)

        assert rxn_dg is None
        assert "B" in missing

    def test_placeholder_value(self):
        """Test handling of placeholder values."""
        metabolite_dg = {"A": -100.0, "B": 10000000.0}  # B has placeholder
        stoichiometry = {"A": -1, "B": 1}

        rxn_dg, missing = compute_reaction_delta_g(metabolite_dg, stoichiometry)

        assert rxn_dg is None
        assert "B" in missing

    def test_default_value(self):
        """Test using default value for missing metabolites."""
        metabolite_dg = {"A": -100.0}  # B is missing
        stoichiometry = {"A": -1, "B": 1}

        rxn_dg, missing = compute_reaction_delta_g(metabolite_dg, stoichiometry, default_delta_g=-50.0)

        # Should use default ΔG° = -50 for B
        # ΔG°_rxn = (-50) - (-100) = +50 kJ/mol
        assert rxn_dg == pytest.approx(50.0)
        assert "B" in missing


class TestMetaboliteDeltaGToReactionKeq:
    """Test conversion from metabolite ΔG° to reaction Keq."""

    def test_simple_network(self):
        """Test simple reaction network."""
        metabolite_dg = {
            "A": -100.0,
            "B": -120.0,  # A -> B: ΔG° = -20 kJ/mol
            "C": -90.0,  # B -> C: ΔG° = +30 kJ/mol
        }

        reactions = [{"id": "R1", "metabolites": {"A": -1, "B": 1}}, {"id": "R2", "metabolites": {"B": -1, "C": 1}}]

        keq_array, info = metabolite_delta_g_to_reaction_keq(metabolite_dg, reactions)

        assert len(keq_array) == 2

        # R1: ΔG° = -20 kJ/mol -> Keq > 1
        assert keq_array[0] > 1.0

        # R2: ΔG° = +30 kJ/mol -> Keq < 1
        assert keq_array[1] < 1.0

        # Check info
        assert info["n_reactions"] == 2
        assert info["reactions_with_complete_data"] == 2
        assert info["coverage_complete"] == 1.0

    def test_missing_data(self):
        """Test handling of missing thermodynamic data."""
        metabolite_dg = {"A": -100.0}  # B is missing

        reactions = [{"id": "R1", "metabolites": {"A": -1, "B": 1}}]

        keq_array, info = metabolite_delta_g_to_reaction_keq(metabolite_dg, reactions, default_keq=5.0)

        assert keq_array[0] == 5.0  # Should use default
        assert info["reactions_with_no_data"] == 1
        assert info["coverage_complete"] == 0.0

    def test_temperature_dependence(self):
        """Test temperature dependence."""
        metabolite_dg = {"A": -100.0, "B": -120.0}  # ΔG° = -20 kJ/mol
        reactions = [{"id": "R1", "metabolites": {"A": -1, "B": 1}}]

        keq_298, _ = metabolite_delta_g_to_reaction_keq(metabolite_dg, reactions, T=298.15)
        keq_373, _ = metabolite_delta_g_to_reaction_keq(metabolite_dg, reactions, T=373.15)

        # Higher temperature should give smaller Keq for negative ΔG°
        assert keq_373[0] < keq_298[0]


class TestValidateThermodynamicConsistency:
    """Test thermodynamic consistency validation."""

    def test_consistent_keq(self):
        """Test thermodynamically consistent Keq values."""
        # Simple cycle: A -> B -> A with consistent Keq
        # If K1 = 2 (A -> B) and K2 = 0.5 (B -> A), then K1 * K2 = 1 ✓

        keq = np.array([2.0, 0.5])
        S = np.array([[-1, 1], [1, -1]])  # A->B, B->A

        is_consistent, violation = validate_thermodynamic_consistency(keq, S)
        assert is_consistent
        assert violation < 1e-10

    def test_inconsistent_keq(self):
        """Test thermodynamically inconsistent Keq values."""
        # Inconsistent cycle: K1 = 10, K2 = 1 (should be K2 = 0.1)
        keq = np.array([10.0, 1.0])
        S = np.array([[-1, 1], [1, -1]])  # A->B, B->A

        is_consistent, violation = validate_thermodynamic_consistency(keq, S)
        assert not is_consistent
        assert violation > 1e-6

    def test_no_conservation_laws(self):
        """Test when there are no conservation laws."""
        # Single irreversible reaction A -> B (no cycles)
        keq = np.array([5.0])
        S = np.array([[-1], [1]])  # A consumed, B produced

        is_consistent, violation = validate_thermodynamic_consistency(keq, S)
        assert is_consistent  # No constraints to violate
        assert violation == 0.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_extreme_delta_g(self):
        """Test extreme ΔG° values."""
        # Very negative ΔG° -> very large Keq
        keq = delta_g_to_keq(-100.0)
        assert keq > 1e40
        assert np.isfinite(keq)

        # Very positive ΔG° -> very small Keq
        keq = delta_g_to_keq(100.0)
        assert keq < 1e-40
        assert keq > 0.0

    def test_zero_stoichiometry(self):
        """Test reaction with zero stoichiometry."""
        metabolite_dg = {"A": -100.0}
        stoichiometry = {"A": 0}  # No net change

        rxn_dg, missing = compute_reaction_delta_g(metabolite_dg, stoichiometry)
        assert rxn_dg == pytest.approx(0.0)
        assert len(missing) == 0

    def test_empty_inputs(self):
        """Test empty inputs."""
        keq_array, info = metabolite_delta_g_to_reaction_keq({}, [])
        assert len(keq_array) == 0
        assert info["n_reactions"] == 0

    def test_warning_on_missing_data(self):
        """Test warnings are issued for missing data."""
        metabolite_dg = {"A": -100.0}  # B missing
        reactions = [{"id": "R1", "metabolites": {"A": -1, "B": 1}}]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            metabolite_delta_g_to_reaction_keq(metabolite_dg, reactions, verbose=True)

            # Should issue warning about missing data
            assert len(w) > 0
            assert any("Missing" in str(warning.message) for warning in w)


if __name__ == "__main__":
    pytest.main([__file__])
