"""
Tests for genome-scale model utilities and performance optimizations.

Tests the GenomeScaleAnalyzer class and related functionality for
handling large-scale metabolic models efficiently.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch
import warnings
import yaml

import numpy as np
import pytest
from scipy import sparse

from llrq import GenomeScaleAnalyzer, load_genome_scale_model, compare_model_sizes
from llrq import from_sbml, SBMLParser, ReactionNetwork
from llrq.genome_scale import GenomeScaleAnalyzer


class TestGenomeScaleAnalyzer:
    """Test GenomeScaleAnalyzer functionality."""

    def get_test_file_path(self, filename):
        """Get path to test data file."""
        return os.path.join(os.path.dirname(__file__), "data", filename)

    def test_lazy_loading(self):
        """Test lazy loading functionality."""
        test_file = self.get_test_file_path("simple_reaction.xml")

        # Test with lazy loading
        analyzer = GenomeScaleAnalyzer(test_file, lazy_load=True)
        assert analyzer._parser is None
        assert analyzer._network_data is None

        # Accessing parser should trigger loading
        parser = analyzer.parser
        assert analyzer._parser is not None
        assert "parser_load_time" in analyzer.performance_metrics

        # Accessing network data should load it
        data = analyzer.network_data
        assert analyzer._network_data is not None
        assert "network_data_time" in analyzer.performance_metrics

    def test_eager_loading(self):
        """Test eager loading functionality."""
        test_file = self.get_test_file_path("simple_reaction.xml")

        # Test without lazy loading
        analyzer = GenomeScaleAnalyzer(test_file, lazy_load=False)
        assert analyzer._parser is not None
        assert analyzer._network_data is not None
        assert "full_load_time" in analyzer.performance_metrics

    def test_model_statistics(self):
        """Test model statistics computation."""
        test_file = self.get_test_file_path("two_reactions.xml")
        analyzer = GenomeScaleAnalyzer(test_file, lazy_load=False)

        stats = analyzer.get_model_statistics()

        # Check basic structure
        assert "n_species" in stats
        assert "n_reactions" in stats
        assert "n_compartments" in stats
        assert "compartments" in stats

        # Check reaction properties
        assert "reversible_reactions" in stats
        assert "irreversible_reactions" in stats

        # Check matrix properties
        assert "matrix_type" in stats
        assert "non_zero_elements" in stats
        assert "sparsity" in stats
        assert "matrix_memory_mb" in stats

        # For this test file, we expect 3 species, 2 reactions
        assert stats["n_species"] == 3
        assert stats["n_reactions"] == 2

    def test_create_network(self):
        """Test network creation with optimal settings."""
        test_file = self.get_test_file_path("simple_reaction.xml")
        analyzer = GenomeScaleAnalyzer(test_file, lazy_load=False)

        # Test network creation
        network = analyzer.create_network()
        assert isinstance(network, ReactionNetwork)
        assert "network_creation_time" in analyzer.performance_metrics

        # Test with explicit sparse setting
        network_sparse = analyzer.create_network(use_sparse=True)
        assert network_sparse.is_sparse

        # Test with explicit dense setting
        network_dense = analyzer.create_network(use_sparse=False)
        assert not network_dense.is_sparse

    def test_compartment_analysis(self):
        """Test compartment analysis functionality."""
        test_file = self.get_test_file_path("two_reactions.xml")
        analyzer = GenomeScaleAnalyzer(test_file, lazy_load=False)

        compartment_analysis = analyzer.get_compartment_analysis()

        # Should have at least one compartment
        assert len(compartment_analysis) >= 1

        # Check structure of compartment data
        for comp_id, comp_data in compartment_analysis.items():
            assert "species" in comp_data
            assert "reactions" in comp_data
            assert "internal_reactions" in comp_data
            assert "transport_reactions" in comp_data
            assert "n_species" in comp_data
            assert "n_reactions" in comp_data
            assert "n_internal_reactions" in comp_data
            assert "n_transport_reactions" in comp_data

    def test_numerical_stability_check(self):
        """Test numerical stability analysis."""
        test_file = self.get_test_file_path("simple_reaction.xml")
        analyzer = GenomeScaleAnalyzer(test_file, lazy_load=False)

        stability = analyzer.check_numerical_stability()

        # Check required fields
        required_fields = [
            "max_stoichiometric_coeff",
            "min_nonzero_stoichiometric_coeff",
            "stoichiometric_range",
            "large_coefficients",
            "small_coefficients",
            "isolated_species",
            "empty_reactions",
            "extreme_flux_bounds",
            "unbounded_reactions",
            "stability_warnings",
        ]

        for field in required_fields:
            assert field in stability

        # For simple test model, should be stable
        assert isinstance(stability["stability_warnings"], list)

    def test_compartment_submodel_extraction(self):
        """Test extracting submodels by compartment."""
        test_file = self.get_test_file_path("two_reactions.xml")
        analyzer = GenomeScaleAnalyzer(test_file, lazy_load=False)

        # Get compartments
        compartment_analysis = analyzer.get_compartment_analysis()
        if not compartment_analysis:
            pytest.skip("No compartments in test model")

        comp_id = list(compartment_analysis.keys())[0]

        # Test single compartment extraction
        submodel = analyzer.extract_compartment_submodel(comp_id)
        assert isinstance(submodel, GenomeScaleAnalyzer)

        sub_stats = submodel.get_model_statistics()
        assert sub_stats["n_species"] > 0
        assert sub_stats["n_reactions"] > 0

        # Test multiple compartments
        if len(compartment_analysis) > 1:
            comp_ids = list(compartment_analysis.keys())[:2]
            submodel_multi = analyzer.extract_compartment_submodel(comp_ids)
            multi_stats = submodel_multi.get_model_statistics()
            assert multi_stats["n_species"] >= sub_stats["n_species"]

    def test_pathway_submodel_extraction(self):
        """Test extracting pathway submodels."""
        test_file = self.get_test_file_path("two_reactions.xml")
        analyzer = GenomeScaleAnalyzer(test_file, lazy_load=False)

        data = analyzer.network_data
        reaction_ids = data["reaction_ids"]

        if len(reaction_ids) > 0:
            # Test single reaction pathway
            submodel = analyzer.extract_pathway_submodel([reaction_ids[0]], include_connected=False)
            assert isinstance(submodel, GenomeScaleAnalyzer)

            sub_stats = submodel.get_model_statistics()
            assert sub_stats["n_reactions"] >= 1

            # Test with connected reactions
            if len(reaction_ids) > 1:
                submodel_connected = analyzer.extract_pathway_submodel([reaction_ids[0]], include_connected=True)
                connected_stats = submodel_connected.get_model_statistics()
                assert connected_stats["n_reactions"] >= sub_stats["n_reactions"]

    def test_empty_submodel_error(self):
        """Test error handling for empty submodels."""
        test_file = self.get_test_file_path("simple_reaction.xml")
        analyzer = GenomeScaleAnalyzer(test_file, lazy_load=False)

        # Test invalid compartment
        with pytest.raises(ValueError, match="No species found in compartments"):
            analyzer.extract_compartment_submodel("nonexistent_compartment")

        # Test empty reaction list
        with pytest.raises(ValueError, match="No reactions selected for submodel"):
            analyzer.extract_pathway_submodel([])

    def test_print_summary(self):
        """Test print summary functionality (basic smoke test)."""
        test_file = self.get_test_file_path("simple_reaction.xml")
        analyzer = GenomeScaleAnalyzer(test_file, lazy_load=False)

        # Should not raise an exception
        try:
            analyzer.print_summary()
        except Exception as e:
            pytest.fail(f"print_summary raised an exception: {e}")


class TestGenomeScaleAnalyzerYAMLSupport:
    """Test GenomeScaleAnalyzer support for YAML files - this would have failed before the fix."""

    def create_test_yaml_content(self):
        """Create test YAML content mimicking yeast-GEM structure."""
        yaml_data = [
            {"metaData": {"id": "test_model", "name": "Test Model", "version": "1.0"}},
            {
                "metabolites": [
                    {
                        "id": "s_001",
                        "name": "Glucose",
                        "compartment": "c",
                        "formula": "C6H12O6",
                        "charge": 0,
                        "deltaG": -915.0,  # kJ/mol
                    },
                    {
                        "id": "s_002",
                        "name": "ATP",
                        "compartment": "c",
                        "formula": "C10H16N5O13P3",
                        "charge": -4,
                        "deltaG": -2292.0,  # kJ/mol
                    },
                ]
            },
            {
                "reactions": [
                    {
                        "id": "r_001",
                        "name": "Glucose transport",
                        "metabolites": {"s_001": -1, "s_002": 1},  # s_001 -> s_002
                        "lower_bound": 0,
                        "upper_bound": 1000,
                        "gene_reaction_rule": "gene1 or gene2",
                        "subsystem": ["Transport"],
                    },
                ]
            },
        ]
        return yaml.dump(yaml_data, default_flow_style=False)

    def test_yaml_file_loading_with_lazy_load(self):
        """Test that GenomeScaleAnalyzer can load YAML files with lazy loading."""
        yaml_content = self.create_test_yaml_content()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            temp_file = f.name

        try:
            # This would have failed before the fix with SBML parsing error
            analyzer = GenomeScaleAnalyzer(temp_file, lazy_load=True)

            # Verify file format detection
            assert analyzer.file_format == "yaml"

            # Verify lazy loading - data should not be loaded yet
            assert analyzer._network_data is None
            assert analyzer._parser is None

            # Access network_data should trigger loading
            data = analyzer.network_data
            assert data is not None
            assert "species_ids" in data
            assert "reaction_ids" in data
            assert "stoichiometric_matrix" in data

            # For YAML files, parser should return None
            assert analyzer.parser is None

        finally:
            import os

            os.unlink(temp_file)

    def test_yaml_file_loading_with_eager_load(self):
        """Test that GenomeScaleAnalyzer can load YAML files with eager loading."""
        yaml_content = self.create_test_yaml_content()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            temp_file = f.name

        try:
            # This would have failed before the fix
            analyzer = GenomeScaleAnalyzer(temp_file, lazy_load=False)

            # Verify data is already loaded
            assert analyzer._network_data is not None
            assert "full_load_time" in analyzer.performance_metrics

            # Verify correct format detection
            assert analyzer.file_format == "yaml"

        finally:
            import os

            os.unlink(temp_file)

    def test_yaml_model_statistics(self):
        """Test model statistics computation for YAML files."""
        yaml_content = self.create_test_yaml_content()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            temp_file = f.name

        try:
            analyzer = GenomeScaleAnalyzer(temp_file, lazy_load=False)
            stats = analyzer.get_model_statistics()

            # Check basic structure - same as SBML files
            assert "n_species" in stats
            assert "n_reactions" in stats
            assert "n_compartments" in stats
            assert "compartments" in stats

            # For our test YAML, we expect 2 species, 1 reaction
            assert stats["n_species"] == 2
            assert stats["n_reactions"] == 1

        finally:
            import os

            os.unlink(temp_file)

    def test_yaml_network_creation(self):
        """Test network creation from YAML files."""
        yaml_content = self.create_test_yaml_content()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            temp_file = f.name

        try:
            analyzer = GenomeScaleAnalyzer(temp_file, lazy_load=False)

            # Test network creation
            network = analyzer.create_network()
            assert isinstance(network, ReactionNetwork)
            assert "network_creation_time" in analyzer.performance_metrics

            # Verify network properties
            assert len(network.species_ids) == 2
            assert len(network.reaction_ids) == 1

        finally:
            import os

            os.unlink(temp_file)

    def test_load_genome_scale_model_yaml(self):
        """Test load_genome_scale_model convenience function with YAML files."""
        yaml_content = self.create_test_yaml_content()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_file = f.name

        try:
            # This would have failed before the fix
            analyzer = load_genome_scale_model(temp_file)
            assert isinstance(analyzer, GenomeScaleAnalyzer)
            assert analyzer.file_format == "yaml"

        finally:
            import os

            os.unlink(temp_file)

    def test_unsupported_file_format(self):
        """Test error handling for unsupported file formats."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some content")
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                GenomeScaleAnalyzer(temp_file)
        finally:
            import os

            os.unlink(temp_file)

    def test_yaml_vs_xml_extension_detection(self):
        """Test that file format detection works correctly for different extensions."""
        yaml_content = self.create_test_yaml_content()

        # Test .yml extension
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            yml_file = f.name

        # Test .yaml extension
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_file = f.name

        try:
            analyzer_yml = GenomeScaleAnalyzer(yml_file, lazy_load=True)
            analyzer_yaml = GenomeScaleAnalyzer(yaml_file, lazy_load=True)

            assert analyzer_yml.file_format == "yaml"
            assert analyzer_yaml.file_format == "yaml"

        finally:
            import os

            os.unlink(yml_file)
            os.unlink(yaml_file)


class TestSparseMatrixIntegration:
    """Test sparse matrix integration across the package."""

    def get_test_file_path(self, filename):
        """Get path to test data file."""
        return os.path.join(os.path.dirname(__file__), "data", filename)

    def test_sparse_matrix_creation(self):
        """Test sparse matrix creation in SBML parser."""
        test_file = self.get_test_file_path("two_reactions.xml")
        parser = SBMLParser(test_file)

        species_info = parser.get_species_info()
        reactions = parser.get_reaction_info()
        species_ids = list(species_info.keys())

        # Test forced sparse creation
        S_sparse = parser.create_stoichiometric_matrix(species_ids, reactions, use_sparse=True)
        assert sparse.issparse(S_sparse)

        # Test forced dense creation
        S_dense = parser.create_stoichiometric_matrix(species_ids, reactions, use_sparse=False)
        assert not sparse.issparse(S_dense)

        # Test auto-detection (should be dense for small matrices)
        S_auto = parser.create_stoichiometric_matrix(species_ids, reactions, use_sparse=None)
        # Small test matrices should default to dense
        assert not sparse.issparse(S_auto)

    def test_reaction_network_sparse_operations(self):
        """Test ReactionNetwork operations with sparse matrices."""
        test_file = self.get_test_file_path("two_reactions.xml")
        parser = SBMLParser(test_file)

        species_info = parser.get_species_info()
        reactions = parser.get_reaction_info()
        species_ids = list(species_info.keys())

        # Create sparse stoichiometric matrix
        S_sparse = parser.create_stoichiometric_matrix(species_ids, reactions, use_sparse=True)

        # Create network with sparse matrix
        network = ReactionNetwork(
            species_ids=species_ids,
            reaction_ids=[r["id"] for r in reactions],
            stoichiometric_matrix=S_sparse,
            species_info=species_info,
            reaction_info=reactions,
            use_sparse=True,
        )

        # Test properties
        assert network.is_sparse
        assert isinstance(network.sparsity, float)
        assert 0 <= network.sparsity <= 1

        # Test matrix operations
        A = network.get_reactant_stoichiometry_matrix()
        B = network.get_product_stoichiometry_matrix()

        assert sparse.issparse(A)
        assert sparse.issparse(B)

        # Test conservation laws
        conservation_laws = network.find_conservation_laws()
        assert isinstance(conservation_laws, np.ndarray)

    def test_auto_sparse_detection(self):
        """Test automatic sparse matrix detection."""
        test_file = self.get_test_file_path("simple_reaction.xml")

        # Create a mock large sparse matrix
        n_species = 1000
        n_reactions = 1500

        # Create a sparse matrix manually
        row_indices = np.random.choice(n_species, size=3000, replace=True)
        col_indices = np.random.choice(n_reactions, size=3000, replace=True)
        data = np.random.randn(3000)

        S_large_sparse = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n_species, n_reactions))

        # Test that auto-detection chooses sparse
        species_ids = [f"s_{i}" for i in range(n_species)]
        reaction_ids = [f"r_{i}" for i in range(n_reactions)]

        network = ReactionNetwork(
            species_ids=species_ids,
            reaction_ids=reaction_ids,
            stoichiometric_matrix=S_large_sparse,
            use_sparse=None,  # Auto-detect
        )

        assert network.is_sparse
        assert network.sparsity > 0.95  # Should be quite sparse


class TestPerformanceOptimizations:
    """Test performance optimizations for large models."""

    def get_test_file_path(self, filename):
        """Get path to test data file."""
        return os.path.join(os.path.dirname(__file__), "data", filename)

    def test_from_sbml_auto_detection(self):
        """Test automatic genome-scale analyzer detection."""
        test_file = self.get_test_file_path("simple_reaction.xml")

        # Mock file size to be large
        with patch("os.path.getsize") as mock_getsize:
            mock_getsize.return_value = 2 * 1024 * 1024  # 2MB

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                network, dynamics, solver, visualizer = from_sbml(test_file)

                # Should have triggered genome-scale path and warnings
                assert len(w) > 0  # Should have warnings about large model

    def test_from_sbml_explicit_genome_scale(self):
        """Test explicit genome-scale analyzer usage."""
        test_file = self.get_test_file_path("simple_reaction.xml")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            network, dynamics, solver, visualizer = from_sbml(test_file, use_genome_scale_analyzer=True)

            # Should work without errors
            assert network is not None
            assert dynamics is not None
            assert solver is not None
            assert visualizer is not None

    def test_from_sbml_explicit_standard(self):
        """Test explicit standard parsing."""
        test_file = self.get_test_file_path("simple_reaction.xml")

        network, dynamics, solver, visualizer = from_sbml(test_file, use_genome_scale_analyzer=False)

        # Should work without errors
        assert network is not None
        assert dynamics is not None
        assert solver is not None
        assert visualizer is not None

    def test_solver_sparse_matrix_handling(self):
        """Test that solver handles sparse matrices correctly."""
        test_file = self.get_test_file_path("simple_reaction.xml")
        parser = SBMLParser(test_file)

        species_info = parser.get_species_info()
        reactions = parser.get_reaction_info()
        species_ids = list(species_info.keys())

        # Create sparse matrix
        S_sparse = parser.create_stoichiometric_matrix(species_ids, reactions, use_sparse=True)

        # Create network with sparse matrix
        network = ReactionNetwork(
            species_ids=species_ids,
            reaction_ids=[r["id"] for r in reactions],
            stoichiometric_matrix=S_sparse,
            species_info=species_info,
            reaction_info=reactions,
            use_sparse=True,
        )

        # Create dynamics and solver
        from llrq import LLRQDynamics, LLRQSolver

        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)

        # Should create without errors
        assert solver._rankS >= 0
        assert solver._B.shape[1] == solver._rankS


class TestFBCExtensions:
    """Test FBC extension parsing and handling."""

    def get_test_file_path(self, filename):
        """Get path to test data file."""
        return os.path.join(os.path.dirname(__file__), "data", filename)

    def test_fbc_objectives_parsing(self):
        """Test FBC objectives parsing."""
        test_file = self.get_test_file_path("simple_reaction.xml")
        parser = SBMLParser(test_file)

        objectives = parser.get_fbc_objectives()

        # Should return proper structure even if no FBC
        assert "active_objective" in objectives
        assert "objectives" in objectives
        assert isinstance(objectives["objectives"], list)

    def test_gene_products_parsing(self):
        """Test gene products parsing."""
        test_file = self.get_test_file_path("simple_reaction.xml")
        parser = SBMLParser(test_file)

        gene_products = parser.get_gene_products()

        # Should return dict even if no gene products
        assert isinstance(gene_products, dict)

    def test_fbc_bounds_in_reactions(self):
        """Test FBC bounds parsing in reactions."""
        test_file = self.get_test_file_path("simple_reaction.xml")
        parser = SBMLParser(test_file)

        reactions = parser.get_reaction_info()

        # Check that FBC fields are included
        for reaction in reactions:
            assert "fbc_bounds" in reaction
            assert "gene_association" in reaction

    def test_network_data_includes_fbc(self):
        """Test that network data extraction includes FBC information."""
        test_file = self.get_test_file_path("simple_reaction.xml")
        parser = SBMLParser(test_file)

        network_data = parser.extract_network_data()

        # Should include FBC fields
        assert "fbc_objectives" in network_data
        assert "gene_products" in network_data


class TestUtilityFunctions:
    """Test utility functions."""

    def get_test_file_path(self, filename):
        """Get path to test data file."""
        return os.path.join(os.path.dirname(__file__), "data", filename)

    def test_load_genome_scale_model(self):
        """Test load_genome_scale_model convenience function."""
        test_file = self.get_test_file_path("simple_reaction.xml")

        # Test with lazy loading (default)
        analyzer = load_genome_scale_model(test_file)
        assert isinstance(analyzer, GenomeScaleAnalyzer)
        assert analyzer.lazy_load

        # Test without lazy loading
        analyzer = load_genome_scale_model(test_file, lazy_load=False)
        assert isinstance(analyzer, GenomeScaleAnalyzer)
        assert not analyzer.lazy_load

    def test_compare_model_sizes(self):
        """Test model size comparison function."""
        test_file1 = self.get_test_file_path("simple_reaction.xml")
        test_file2 = self.get_test_file_path("two_reactions.xml")

        models = {"simple": test_file1, "two_reactions": test_file2}

        comparison = compare_model_sizes(models)

        assert isinstance(comparison, dict)
        assert "simple" in comparison
        assert "two_reactions" in comparison

        # Should have model statistics
        for model_name, stats in comparison.items():
            if "error" not in stats:  # Skip error cases
                assert "n_species" in stats
                assert "n_reactions" in stats

    def test_compare_model_sizes_with_error(self):
        """Test model size comparison with invalid files."""
        models = {"valid": self.get_test_file_path("simple_reaction.xml"), "invalid": "nonexistent_file.xml"}

        comparison = compare_model_sizes(models)

        assert "valid" in comparison
        assert "invalid" in comparison

        # Valid model should have stats
        assert "n_species" in comparison["valid"]

        # Invalid model should have error
        assert "error" in comparison["invalid"]


class TestErrorHandling:
    """Test error handling in genome-scale functionality."""

    def test_invalid_sbml_file(self):
        """Test handling of invalid SBML files."""
        with pytest.raises((FileNotFoundError, Exception)):
            GenomeScaleAnalyzer("nonexistent_file.xml")

    def test_empty_submodel_extraction(self):
        """Test error handling for empty submodel extractions."""
        # Create a minimal analyzer with mock data
        analyzer = GenomeScaleAnalyzer.__new__(GenomeScaleAnalyzer)
        analyzer.model_file = "test"
        analyzer.file_format = "sbml"
        analyzer.lazy_load = False
        analyzer._parser = None
        analyzer._network_data = {
            "species": {},
            "reactions": [],
            "parameters": {},
            "stoichiometric_matrix": np.array([]).reshape(0, 0),
            "species_ids": [],
            "reaction_ids": [],
            "fbc_objectives": {},
            "gene_products": {},
        }
        analyzer._network = None
        analyzer.performance_metrics = {}
        analyzer._model_stats = None

        # Should raise appropriate errors
        with pytest.raises(ValueError):
            analyzer.extract_compartment_submodel("nonexistent")

        with pytest.raises(ValueError):
            analyzer.extract_pathway_submodel([])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
