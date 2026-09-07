import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "smoke.py"
SPEC = importlib.util.spec_from_file_location("smoke", SCRIPT)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def complete_record():
    return {
        "beta": 0.2,
        "u": -0.3,
        "c": 0.1,
        "f": -1.2,
        "magnetization": 0.4,
        "max_bond": 2,
    }


class ValidateResultTests(unittest.TestCase):
    def test_accepts_a_complete_finite_record(self):
        smoke.validate_result({"records": [complete_record()]})

    def test_rejects_missing_records(self):
        with self.assertRaisesRegex(AssertionError, "missing observations"):
            smoke.validate_result({"records": []})

    def test_rejects_a_missing_observable(self):
        row = complete_record()
        del row["f"]
        with self.assertRaises(KeyError):
            smoke.validate_result({"records": [row]})

    def test_rejects_each_nonfinite_observable(self):
        for field in ("beta", "u", "c", "f", "magnetization"):
            with self.subTest(field=field):
                row = complete_record()
                row[field] = float("nan")
                with self.assertRaisesRegex(AssertionError, field):
                    smoke.validate_result({"records": [row]})


class SmokeCommandTests(unittest.TestCase):
    def make_solver(self, directory, body):
        solver = Path(directory) / "solver"
        solver.write_text("#!/bin/sh\n" + body, encoding="utf-8")
        solver.chmod(solver.stat().st_mode | stat.S_IXUSR)
        return solver.resolve()

    def test_existing_destination_is_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "existing"
            destination.mkdir()
            sentinel = destination / "sentinel"
            sentinel.write_text("keep", encoding="utf-8")
            solver = self.make_solver(temporary, "exit 0\n")

            with self.assertRaises(FileExistsError):
                smoke.run_smoke(solver, destination)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            self.assertEqual(list(destination.iterdir()), [sentinel])

    def test_nonzero_solver_exit_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            solver = self.make_solver(temporary, "exit 7\n")
            destination = Path(temporary) / "new"

            with self.assertRaises(Exception) as raised:
                smoke.run_smoke(solver, destination)

            self.assertEqual(getattr(raised.exception, "returncode", None), 7)


if __name__ == "__main__":
    unittest.main()
