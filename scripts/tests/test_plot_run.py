import importlib.util
from pathlib import Path
import unittest


class PlotFreeEnergyTests(unittest.TestCase):
    def load_plot(self):
        spec = importlib.util.spec_from_file_location(
            "plot_run", Path(__file__).resolve().parents[1] / "plot_run.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_initial_null_f_uses_stored_limit_and_exact_uses_local_dimension(self):
        plot = self.load_plot()
        row = {"beta": 0.0, "f": None, "beta_f": -1.0986122886681098,
               "exact": {"f": None}}
        self.assertEqual(plot.dimensionless_free_energy(row, 3), -1.0986122886681098)
        self.assertEqual(plot.dimensionless_free_energy(row, 3, exact=True), -1.0986122886681098)

    def test_legacy_positive_beta_and_exact_references_remain_plottable(self):
        plot = self.load_plot()
        row = {"beta": 0.5, "f": -1.5, "exact": {"f": -1.6}}
        self.assertEqual(plot.dimensionless_free_energy(row, 2), -0.75)
        self.assertEqual(plot.dimensionless_free_energy(row, 2, exact=True), -0.8)


if __name__ == "__main__":
    unittest.main()
