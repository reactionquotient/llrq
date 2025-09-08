#!/usr/bin/env python3
"""
Comprehensive test to compare reliability of concentration reconstruction algorithms.
"""

import numpy as np
import time
import warnings
import sys

sys.path.insert(0, "src")

from llrq.reaction_network import ReactionNetwork
from llrq.utils.concentration_utils import compute_concentrations_from_quotients


class AlgorithmTester:
    def __init__(self):
        self.results = {"convex": [], "fsolve": []}

    def test_network(self, network, c0, Q_targets, test_name=""):
        """Test both algorithms on a network with given initial conditions and target quotients."""
        print(f"\n=== Testing: {test_name} ===")
        print(f"Initial concentrations: {c0}")
        print(f"Target quotients: {Q_targets}")

        # Create simple B matrix (identity for full rank case)
        n_reactions = len(Q_targets)
        B = np.eye(n_reactions)
        lnKeq = np.zeros(n_reactions)  # Assume K_eq = 1 for simplicity

        for alg in ["convex", "fsolve"]:
            result = self._test_single_algorithm(network, c0, Q_targets, B, lnKeq, alg)
            self.results[alg].append((test_name, result))
            self._print_result(alg, result)

    def _test_single_algorithm(self, network, c0, Q_targets, B, lnKeq, algorithm):
        """Test a single algorithm and return comprehensive metrics."""
        result = {
            "success": False,
            "conservation_error": np.inf,
            "quotient_error": np.inf,
            "computation_time": np.inf,
            "warnings_count": 0,
            "final_concentrations": None,
            "error_message": None,
        }

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            try:
                start_time = time.time()
                c_result = compute_concentrations_from_quotients(
                    Q_targets, c0, network, B, lnKeq, enforce_conservation=True, algorithm=algorithm
                )
                result["computation_time"] = time.time() - start_time
                result["warnings_count"] = len(w)

                if c_result is not None:
                    result["final_concentrations"] = c_result
                    result["success"] = True

                    # Check conservation laws
                    C = network.find_conservation_laws()
                    if C.shape[0] > 0:
                        cons0 = network.compute_conserved_quantities(c0)
                        cons_final = C @ c_result
                        result["conservation_error"] = np.linalg.norm(cons_final - cons0, np.inf)
                    else:
                        result["conservation_error"] = 0.0

                    # Check quotient accuracy
                    Q_computed = network.compute_reaction_quotients(c_result)
                    result["quotient_error"] = np.linalg.norm(Q_computed - Q_targets, np.inf)

            except Exception as e:
                result["error_message"] = str(e)
                if w:
                    result["warnings_count"] = len(w)

        return result

    def _print_result(self, algorithm, result):
        """Print results for a single algorithm."""
        if result["success"]:
            print(
                f"{algorithm:8s}: ✓ Success | Cons Error: {result['conservation_error']:.2e} | "
                f"Quot Error: {result['quotient_error']:.2e} | Time: {result['computation_time']:.4f}s | "
                f"Warnings: {result['warnings_count']}"
            )
        else:
            print(f"{algorithm:8s}: ✗ Failed  | Error: {result['error_message']} | Warnings: {result['warnings_count']}")

    def run_comprehensive_tests(self):
        """Run a comprehensive suite of tests."""
        print("Starting comprehensive algorithm reliability tests...\n")

        # Test 1: Simple reversible reaction A ⇌ B
        network1 = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        self.test_network(network1, np.array([2.0, 1.0]), np.array([0.5]), "Simple A ⇌ B")

        # Test 2: Bimolecular reaction A + B ⇌ C
        network2 = ReactionNetwork(["A", "B", "C"], ["R1"], np.array([[-1], [-1], [1]]))
        self.test_network(network2, np.array([2.0, 2.0, 0.1]), np.array([0.5]), "Bimolecular A + B ⇌ C")

        # Test 3: Two-step pathway A ⇌ B ⇌ C
        network3 = ReactionNetwork(["A", "B", "C"], ["R1", "R2"], np.array([[-1, 0], [1, -1], [0, 1]]))
        self.test_network(network3, np.array([3.0, 1.0, 0.5]), np.array([2.0, 0.25]), "Two-step A ⇌ B ⇌ C")

        # Test 4: Very small concentrations (numerical challenge)
        self.test_network(network1, np.array([1e-8, 1e-9]), np.array([2.0]), "Small concentrations")

        # Test 5: Large concentration ratios
        self.test_network(network1, np.array([1000.0, 0.001]), np.array([1e-6]), "Large concentration ratios")

        # Test 6: Near-equilibrium conditions
        self.test_network(network2, np.array([1.0, 1.0, 1.0]), np.array([1.0]), "Near equilibrium")

        # Test 7: Far from equilibrium
        self.test_network(network2, np.array([10.0, 10.0, 0.001]), np.array([0.0001]), "Far from equilibrium")

    def summarize_results(self):
        """Print summary comparison of algorithms."""
        print("\n" + "=" * 80)
        print("ALGORITHM RELIABILITY SUMMARY")
        print("=" * 80)

        for alg in ["convex", "fsolve"]:
            results = self.results[alg]
            success_count = sum(1 for _, r in results if r["success"])
            total_tests = len(results)

            if success_count > 0:
                successful_results = [r for _, r in results if r["success"]]
                avg_cons_error = np.mean([r["conservation_error"] for r in successful_results])
                avg_quot_error = np.mean([r["quotient_error"] for r in successful_results])
                avg_time = np.mean([r["computation_time"] for r in successful_results])
                total_warnings = sum(r["warnings_count"] for r in successful_results)
            else:
                avg_cons_error = avg_quot_error = avg_time = float("inf")
                total_warnings = sum(r["warnings_count"] for _, r in results)

            print(f"\n{alg.upper()} Algorithm:")
            print(f"  Success Rate: {success_count}/{total_tests} ({100*success_count/total_tests:.1f}%)")
            print(f"  Avg Conservation Error: {avg_cons_error:.2e}")
            print(f"  Avg Quotient Error: {avg_quot_error:.2e}")
            print(f"  Avg Computation Time: {avg_time:.4f}s")
            print(f"  Total Warnings: {total_warnings}")

        # Recommendation
        convex_success = sum(1 for _, r in self.results["convex"] if r["success"])
        fsolve_success = sum(1 for _, r in self.results["fsolve"] if r["success"])

        print(f"\nRECOMMENDATION:")
        if convex_success > fsolve_success:
            print("  Use 'convex' algorithm (higher success rate)")
        elif fsolve_success > convex_success:
            print("  Use 'fsolve' algorithm (higher success rate)")
        else:
            # Compare other metrics for successful cases
            if convex_success > 0 and fsolve_success > 0:
                convex_avg_error = np.mean(
                    [r["conservation_error"] + r["quotient_error"] for _, r in self.results["convex"] if r["success"]]
                )
                fsolve_avg_error = np.mean(
                    [r["conservation_error"] + r["quotient_error"] for _, r in self.results["fsolve"] if r["success"]]
                )

                if convex_avg_error < fsolve_avg_error:
                    print("  Use 'convex' algorithm (lower average error)")
                else:
                    print("  Use 'fsolve' algorithm (lower average error)")
            else:
                print("  Both algorithms have similar performance - use 'convex' as default")


def main():
    tester = AlgorithmTester()
    tester.run_comprehensive_tests()
    tester.summarize_results()


if __name__ == "__main__":
    main()
