"""
Comprehensive tests for LLRQVisualizer class.

Tests plot generation, phase space plotting, stability analysis visualization,
and other visualization features with mock matplotlib to avoid display issues.
"""

import os
import sys
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add source directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# Mock matplotlib to avoid display and import issues in testing
@patch.dict("sys.modules", {"matplotlib": MagicMock(), "matplotlib.pyplot": MagicMock()})
def test_imports():
    """Test that visualization module can be imported with mocked matplotlib."""
    from llrq.llrq_dynamics import LLRQDynamics
    from llrq.reaction_network import ReactionNetwork
    from llrq.solver import LLRQSolver
    from llrq.visualization import LLRQVisualizer

    return ReactionNetwork, LLRQDynamics, LLRQSolver, LLRQVisualizer


class TestLLRQVisualizerInitialization:
    """Test LLRQVisualizer initialization."""

    @patch.dict("sys.modules", {"matplotlib": MagicMock(), "matplotlib.pyplot": MagicMock()})
    def test_initialization_basic(self):
        """Test basic visualizer initialization."""
        from llrq.llrq_dynamics import LLRQDynamics
        from llrq.reaction_network import ReactionNetwork
        from llrq.solver import LLRQSolver
        from llrq.visualization import LLRQVisualizer

        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)
        visualizer = LLRQVisualizer(solver)

        assert visualizer.solver == solver
        assert visualizer.dynamics == dynamics
        assert visualizer.network == network

    @patch.dict("sys.modules", {"matplotlib": MagicMock(), "matplotlib.pyplot": MagicMock()})
    def test_initialization_with_complex_system(self):
        """Test initialization with multi-reaction system."""
        from llrq.llrq_dynamics import LLRQDynamics
        from llrq.reaction_network import ReactionNetwork
        from llrq.solver import LLRQSolver
        from llrq.visualization import LLRQVisualizer

        network = ReactionNetwork(["A", "B", "C"], ["R1", "R2"], np.array([[-1, 0], [1, -1], [0, 1]]))
        K = np.array([[2.0, -0.5], [-0.5, 1.0]])
        dynamics = LLRQDynamics(network, relaxation_matrix=K)
        solver = LLRQSolver(dynamics)
        visualizer = LLRQVisualizer(solver)

        assert visualizer.solver == solver
        assert visualizer.dynamics == dynamics
        assert visualizer.network == network


def create_mock_axes():
    """Helper to create mock matplotlib axes that support both indexing styles."""

    class MockAxes:
        def __init__(self):
            self._axes = [[MagicMock(), MagicMock()], [MagicMock(), MagicMock()]]

        def __getitem__(self, key):
            if isinstance(key, tuple):
                i, j = key
                return self._axes[i][j]
            else:
                return self._axes[key]

    return MockAxes()


class TestPlotDynamics:
    """Test dynamics plotting functionality."""

    @patch.dict("sys.modules", {"matplotlib": MagicMock(), "matplotlib.pyplot": MagicMock()})
    def test_plot_dynamics_basic(self):
        """Test basic dynamics plotting."""
        from llrq.llrq_dynamics import LLRQDynamics
        from llrq.reaction_network import ReactionNetwork
        from llrq.solver import LLRQSolver
        from llrq.visualization import LLRQVisualizer

        # Create mock solution data
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)
        visualizer = LLRQVisualizer(solver)

        # Mock solution data
        t = np.linspace(0, 2, 100)
        solution = {
            "time": t,
            "concentrations": np.column_stack([2 * np.exp(-t), 1 + np.exp(-t)]),
            "reaction_quotients": np.column_stack([0.5 + 0.5 * np.exp(-t)]),
            "log_deviations": np.column_stack([np.log(0.5 + 0.5 * np.exp(-t))]),
            "initial_concentrations": np.array([2.0, 1.0]),
            "success": True,
            "method": "numerical",
        }

        # Mock matplotlib
        with patch("llrq.visualization.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_axes = create_mock_axes()
            mock_plt.subplots.return_value = (mock_fig, mock_axes)

            fig = visualizer.plot_dynamics(solution)

            # Should have called subplots to create figure
            mock_plt.subplots.assert_called_once()
            assert fig == mock_fig

    @patch.dict("sys.modules", {"matplotlib": MagicMock(), "matplotlib.pyplot": MagicMock()})
    def test_plot_dynamics_with_species_selection(self):
        """Test plotting with specific species selection."""
        from llrq.llrq_dynamics import LLRQDynamics
        from llrq.reaction_network import ReactionNetwork
        from llrq.solver import LLRQSolver
        from llrq.visualization import LLRQVisualizer

        network = ReactionNetwork(["A", "B", "C"], ["R1", "R2"], np.array([[-1, 0], [1, -1], [0, 1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)
        visualizer = LLRQVisualizer(solver)

        # Mock solution data
        t = np.linspace(0, 2, 50)
        solution = {
            "time": t,
            "concentrations": np.random.random((50, 3)),
            "reaction_quotients": np.random.random((50, 2)),
            "log_deviations": np.random.random((50, 2)),
            "initial_concentrations": np.array([1.0, 1.0, 1.0]),
            "success": True,
            "method": "numerical",
        }

        with patch("llrq.visualization.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_axes = create_mock_axes()
            mock_plt.subplots.return_value = (mock_fig, mock_axes)

            fig = visualizer.plot_dynamics(solution, species_to_plot=["A", "B"])

            mock_plt.subplots.assert_called_once()

    @patch.dict("sys.modules", {"matplotlib": MagicMock(), "matplotlib.pyplot": MagicMock()})
    def test_plot_dynamics_with_reaction_selection(self):
        """Test plotting with specific reaction selection."""
        from llrq.llrq_dynamics import LLRQDynamics
        from llrq.reaction_network import ReactionNetwork
        from llrq.solver import LLRQSolver
        from llrq.visualization import LLRQVisualizer

        network = ReactionNetwork(["A", "B", "C"], ["forward", "reverse"], np.array([[-1, 0], [1, -1], [0, 1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)
        visualizer = LLRQVisualizer(solver)

        # Mock solution data
        t = np.linspace(0, 1, 50)
        solution = {
            "time": t,
            "concentrations": np.random.random((50, 3)),
            "reaction_quotients": np.random.random((50, 2)),
            "log_deviations": np.random.random((50, 2)),
            "initial_concentrations": np.array([2.0, 1.0, 0.5]),
            "success": True,
            "method": "numerical",
        }

        with patch("llrq.visualization.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_axes = create_mock_axes()
            mock_plt.subplots.return_value = (mock_fig, mock_axes)

            fig = visualizer.plot_dynamics(solution, reactions_to_plot=["forward"])

            mock_plt.subplots.assert_called_once()

    @patch.dict("sys.modules", {"matplotlib": MagicMock(), "matplotlib.pyplot": MagicMock()})
    def test_plot_dynamics_log_scale(self):
        """Test plotting with log scale."""
        from llrq.llrq_dynamics import LLRQDynamics
        from llrq.reaction_network import ReactionNetwork
        from llrq.solver import LLRQSolver
        from llrq.visualization import LLRQVisualizer

        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)
        visualizer = LLRQVisualizer(solver)

        # Mock solution data
        t = np.linspace(0, 2, 50)
        solution = {
            "time": t,
            "concentrations": np.column_stack([2 * np.exp(-t), 1 + np.exp(-t)]),
            "reaction_quotients": np.column_stack([0.5 + 0.5 * np.exp(-t)]),
            "log_deviations": np.column_stack([np.log(0.5 + 0.5 * np.exp(-t))]),
            "initial_concentrations": np.array([2.0, 1.0]),
            "success": True,
            "method": "numerical",
        }

        with patch("llrq.visualization.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_axes = create_mock_axes()
            mock_plt.subplots.return_value = (mock_fig, mock_axes)

            fig = visualizer.plot_dynamics(solution, log_scale=True)

            mock_plt.subplots.assert_called_once()
            # Should have called set_yscale('log') on concentration plot
            mock_axes[0, 0].set_yscale.assert_called_with("log")

    @patch.dict("sys.modules", {"matplotlib": MagicMock(), "matplotlib.pyplot": MagicMock()})
    def test_plot_dynamics_no_concentrations(self):
        """Test plotting when concentrations are None."""
        from llrq.llrq_dynamics import LLRQDynamics
        from llrq.reaction_network import ReactionNetwork
        from llrq.solver import LLRQSolver
        from llrq.visualization import LLRQVisualizer

        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)
        visualizer = LLRQVisualizer(solver)

        # Mock solution data with no concentrations
        t = np.linspace(0, 2, 50)
        solution = {
            "time": t,
            "concentrations": None,  # No concentrations available
            "reaction_quotients": np.column_stack([0.5 + 0.5 * np.exp(-t)]),
            "log_deviations": np.column_stack([np.log(0.5 + 0.5 * np.exp(-t))]),
            "initial_concentrations": np.array([2.0, 1.0]),
            "success": True,
            "method": "numerical",
        }

        with patch("llrq.visualization.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_axes = create_mock_axes()
            mock_plt.subplots.return_value = (mock_fig, mock_axes)

            fig = visualizer.plot_dynamics(solution)

            # Should still create plot but with message about no concentrations
            mock_plt.subplots.assert_called_once()
            mock_axes[0, 0].text.assert_called()  # Should show "not available" message


class TestPhaseSpacePlotting:
    """Test phase space and 2D plotting functionality."""

    @patch.dict("sys.modules", {"matplotlib": MagicMock(), "matplotlib.pyplot": MagicMock()})
    def test_plot_phase_space_2d(self):
        """Test 2D phase space plotting."""
        from llrq.llrq_dynamics import LLRQDynamics
        from llrq.reaction_network import ReactionNetwork
        from llrq.solver import LLRQSolver
        from llrq.visualization import LLRQVisualizer

        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)
        visualizer = LLRQVisualizer(solver)

        # Mock solution data
        t = np.linspace(0, 2, 50)
        solution = {
            "time": t,
            "concentrations": np.column_stack([2 * np.exp(-t), 1 + np.exp(-t)]),
            "reaction_quotients": np.column_stack([0.5 + 0.5 * np.exp(-t)]),
            "log_deviations": np.column_stack([np.log(0.5 + 0.5 * np.exp(-t))]),
            "initial_concentrations": np.array([2.0, 1.0]),
            "success": True,
            "method": "numerical",
        }

        with patch("llrq.visualization.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)

            # Mock the method (if it exists)
            if hasattr(visualizer, "plot_phase_space"):
                fig = visualizer.plot_phase_space(solution, species=["A", "B"])
                mock_plt.subplots.assert_called_once()

    @patch.dict("sys.modules", {"matplotlib": MagicMock(), "matplotlib.pyplot": MagicMock()})
    def test_plot_reaction_quotient_space(self):
        """Test reaction quotient space plotting."""
        from llrq.llrq_dynamics import LLRQDynamics
        from llrq.reaction_network import ReactionNetwork
        from llrq.solver import LLRQSolver
        from llrq.visualization import LLRQVisualizer

        network = ReactionNetwork(["A", "B", "C"], ["R1", "R2"], np.array([[-1, 0], [1, -1], [0, 1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)
        visualizer = LLRQVisualizer(solver)

        # Mock solution data
        t = np.linspace(0, 2, 50)
        solution = {
            "time": t,
            "concentrations": np.random.random((50, 3)),
            "reaction_quotients": np.random.random((50, 2)),
            "log_deviations": np.random.random((50, 2)),
            "initial_concentrations": np.array([2.0, 1.0, 0.5]),
            "success": True,
            "method": "numerical",
        }

        with patch("llrq.visualization.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)

            # Mock the method (if it exists)
            if hasattr(visualizer, "plot_reaction_quotient_space"):
                fig = visualizer.plot_reaction_quotient_space(solution)
                mock_plt.subplots.assert_called_once()


class TestStabilityAnalysis:
    """Test stability analysis visualization."""

    @patch.dict("sys.modules", {"matplotlib": MagicMock(), "matplotlib.pyplot": MagicMock()})
    def test_plot_eigenvalue_spectrum(self):
        """Test eigenvalue spectrum plotting."""
        from llrq.llrq_dynamics import LLRQDynamics
        from llrq.reaction_network import ReactionNetwork
        from llrq.solver import LLRQSolver
        from llrq.visualization import LLRQVisualizer

        network = ReactionNetwork(["A", "B", "C"], ["R1", "R2"], np.array([[-1, 0], [1, -1], [0, 1]]))
        K = np.array([[2.0, -0.1], [-0.1, 1.5]])
        dynamics = LLRQDynamics(network, relaxation_matrix=K)
        solver = LLRQSolver(dynamics)
        visualizer = LLRQVisualizer(solver)

        with patch("llrq.visualization.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)

            # Mock the method (if it exists)
            if hasattr(visualizer, "plot_eigenvalue_spectrum"):
                fig = visualizer.plot_eigenvalue_spectrum()
                mock_plt.subplots.assert_called_once()

    @patch.dict("sys.modules", {"matplotlib": MagicMock(), "matplotlib.pyplot": MagicMock()})
    def test_plot_relaxation_modes(self):
        """Test relaxation mode plotting."""
        from llrq.llrq_dynamics import LLRQDynamics
        from llrq.reaction_network import ReactionNetwork
        from llrq.solver import LLRQSolver
        from llrq.visualization import LLRQVisualizer

        network = ReactionNetwork(["A", "B", "C"], ["R1", "R2"], np.array([[-1, 0], [1, -1], [0, 1]]))
        K = np.array([[1.0, 0.2], [0.2, 2.0]])
        dynamics = LLRQDynamics(network, relaxation_matrix=K)
        solver = LLRQSolver(dynamics)
        visualizer = LLRQVisualizer(solver)

        with patch("llrq.visualization.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_axes = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_axes)

            # Mock the method (if it exists)
            if hasattr(visualizer, "plot_relaxation_modes"):
                fig = visualizer.plot_relaxation_modes()
                mock_plt.subplots.assert_called_once()


class TestNetworkVisualization:
    """Test reaction network visualization."""

    @patch.dict("sys.modules", {"matplotlib": MagicMock(), "matplotlib.pyplot": MagicMock(), "networkx": MagicMock()})
    def test_plot_reaction_network(self):
        """Test reaction network graph plotting."""
        from llrq.llrq_dynamics import LLRQDynamics
        from llrq.reaction_network import ReactionNetwork
        from llrq.solver import LLRQSolver
        from llrq.visualization import LLRQVisualizer

        network = ReactionNetwork(["A", "B", "C"], ["R1", "R2"], np.array([[-1, 0], [1, -1], [0, 1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)
        visualizer = LLRQVisualizer(solver)

        with patch("llrq.visualization.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)

            # Mock the method (if it exists)
            if hasattr(visualizer, "plot_network_graph"):
                fig = visualizer.plot_network_graph()
                mock_plt.subplots.assert_called_once()

    @patch.dict("sys.modules", {"matplotlib": MagicMock(), "matplotlib.pyplot": MagicMock()})
    def test_plot_stoichiometric_matrix(self):
        """Test stoichiometric matrix heatmap."""
        from llrq.llrq_dynamics import LLRQDynamics
        from llrq.reaction_network import ReactionNetwork
        from llrq.solver import LLRQSolver
        from llrq.visualization import LLRQVisualizer

        network = ReactionNetwork(["A", "B", "C"], ["R1", "R2"], np.array([[-1, 0], [1, -1], [0, 1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)
        visualizer = LLRQVisualizer(solver)

        with patch("llrq.visualization.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)

            # Mock the method (if it exists)
            if hasattr(visualizer, "plot_stoichiometric_matrix"):
                fig = visualizer.plot_stoichiometric_matrix()
                mock_plt.subplots.assert_called_once()


class TestErrorHandling:
    """Test error handling in visualization."""

    @patch.dict("sys.modules", {"matplotlib": MagicMock(), "matplotlib.pyplot": MagicMock()})
    def test_plot_empty_solution(self):
        """Test plotting with empty solution data."""
        from llrq.llrq_dynamics import LLRQDynamics
        from llrq.reaction_network import ReactionNetwork
        from llrq.solver import LLRQSolver
        from llrq.visualization import LLRQVisualizer

        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)
        visualizer = LLRQVisualizer(solver)

        # Empty solution
        solution = {
            "time": np.array([]),
            "concentrations": None,
            "reaction_quotients": np.array([]).reshape(0, 1),
            "log_deviations": np.array([]).reshape(0, 1),
            "initial_concentrations": np.array([1.0, 1.0]),
            "success": False,
            "method": "numerical",
        }

        with patch("llrq.visualization.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_axes = create_mock_axes()
            mock_plt.subplots.return_value = (mock_fig, mock_axes)

            # Should handle gracefully
            try:
                fig = visualizer.plot_dynamics(solution)
                # If it succeeds, that's good
            except (ValueError, IndexError):
                # It's also acceptable to raise an error for empty data
                pass

    @patch.dict("sys.modules", {"matplotlib": MagicMock(), "matplotlib.pyplot": MagicMock()})
    def test_plot_invalid_species_selection(self):
        """Test plotting with invalid species selection."""
        from llrq.llrq_dynamics import LLRQDynamics
        from llrq.reaction_network import ReactionNetwork
        from llrq.solver import LLRQSolver
        from llrq.visualization import LLRQVisualizer

        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)
        visualizer = LLRQVisualizer(solver)

        # Mock solution data
        t = np.linspace(0, 2, 50)
        solution = {
            "time": t,
            "concentrations": np.column_stack([2 * np.exp(-t), 1 + np.exp(-t)]),
            "reaction_quotients": np.column_stack([0.5 + 0.5 * np.exp(-t)]),
            "log_deviations": np.column_stack([np.log(0.5 + 0.5 * np.exp(-t))]),
            "initial_concentrations": np.array([2.0, 1.0]),
            "success": True,
            "method": "numerical",
        }

        with patch("llrq.visualization.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_axes = create_mock_axes()
            mock_plt.subplots.return_value = (mock_fig, mock_axes)

            # Should handle invalid species gracefully
            fig = visualizer.plot_dynamics(solution, species_to_plot=["C", "D"])  # Invalid
            mock_plt.subplots.assert_called_once()

    @patch.dict("sys.modules", {"matplotlib": MagicMock(), "matplotlib.pyplot": MagicMock()})
    def test_matplotlib_style_fallback(self):
        """Test fallback when matplotlib styles are not available."""
        from llrq.llrq_dynamics import LLRQDynamics
        from llrq.reaction_network import ReactionNetwork
        from llrq.solver import LLRQSolver

        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)

        # Mock plt.style.use to raise OSError (style not found)
        with patch("llrq.visualization.plt") as mock_plt:
            mock_plt.style.use.side_effect = [OSError, OSError, None]  # Fail twice, succeed on default

            from llrq.visualization import LLRQVisualizer

            visualizer = LLRQVisualizer(solver)

            # Should have tried multiple styles and fallen back to default
            assert mock_plt.style.use.call_count >= 2


class TestCustomPlotParameters:
    """Test custom plot parameters and options."""

    @patch.dict("sys.modules", {"matplotlib": MagicMock(), "matplotlib.pyplot": MagicMock()})
    def test_plot_dynamics_custom_figsize(self):
        """Test plotting with custom figure size."""
        from llrq.llrq_dynamics import LLRQDynamics
        from llrq.reaction_network import ReactionNetwork
        from llrq.solver import LLRQSolver
        from llrq.visualization import LLRQVisualizer

        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)
        visualizer = LLRQVisualizer(solver)

        # Mock solution data
        t = np.linspace(0, 2, 50)
        solution = {
            "time": t,
            "concentrations": np.column_stack([2 * np.exp(-t), 1 + np.exp(-t)]),
            "reaction_quotients": np.column_stack([0.5 + 0.5 * np.exp(-t)]),
            "log_deviations": np.column_stack([np.log(0.5 + 0.5 * np.exp(-t))]),
            "initial_concentrations": np.array([2.0, 1.0]),
            "success": True,
            "method": "numerical",
        }

        with patch("llrq.visualization.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_axes = create_mock_axes()
            mock_plt.subplots.return_value = (mock_fig, mock_axes)

            fig = visualizer.plot_dynamics(solution, figsize=(10, 6))

            # Should have called subplots with custom figsize
            mock_plt.subplots.assert_called_with(2, 2, figsize=(10, 6))

    @patch.dict("sys.modules", {"matplotlib": MagicMock(), "matplotlib.pyplot": MagicMock()})
    def test_plot_dynamics_mass_action_comparison(self):
        """Test plotting with mass action comparison."""
        from llrq.llrq_dynamics import LLRQDynamics
        from llrq.reaction_network import ReactionNetwork
        from llrq.solver import LLRQSolver
        from llrq.visualization import LLRQVisualizer

        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)
        visualizer = LLRQVisualizer(solver)

        # Mock solution data
        t = np.linspace(0, 2, 50)
        solution = {
            "time": t,
            "concentrations": np.column_stack([2 * np.exp(-t), 1 + np.exp(-t)]),
            "reaction_quotients": np.column_stack([0.5 + 0.5 * np.exp(-t)]),
            "log_deviations": np.column_stack([np.log(0.5 + 0.5 * np.exp(-t))]),
            "initial_concentrations": np.array([2.0, 1.0]),
            "success": True,
            "method": "numerical",
        }

        with patch("llrq.visualization.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_axes = create_mock_axes()
            mock_plt.subplots.return_value = (mock_fig, mock_axes)

            fig = visualizer.plot_dynamics(solution, compare_mass_action=True)

            mock_plt.subplots.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
