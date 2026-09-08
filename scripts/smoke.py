#!/usr/bin/env python3
"""Run short real and complex solver cases into a newly reserved directory."""

import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def validate_result(result):
    assert result["records"], "missing observations"
    for row in result["records"]:
        for field in ("beta", "u", "c", "magnetization"):
            value = row[field]
            assert isinstance(value, (int, float)) and not isinstance(value, bool)
            assert math.isfinite(value), field
        assert row["beta"] >= 0.0, "beta"
        if row["beta"] == 0.0:
            assert row["f"] is None, "f must be null at beta=0"
            beta_f = row["beta_f"]
        else:
            f = row["f"]
            assert isinstance(f, (int, float)) and not isinstance(f, bool), "f"
            assert math.isfinite(f), "f"
            beta_f = row.get("beta_f", row["beta"] * f)
        assert isinstance(beta_f, (int, float)) and not isinstance(beta_f, bool), "beta_f"
        assert math.isfinite(beta_f), "beta_f"
        if row["beta"] > 0.0:
            assert math.isclose(beta_f, row["beta"] * row["f"], rel_tol=2e-15, abs_tol=2e-15), "beta_f"
        assert row["max_bond"] >= 1


def _write_tfim_config(config_path, output_path):
    source = (ROOT / "configs" / "quickstart.toml").read_text(encoding="utf-8")
    # JSON basic-string escapes are also TOML escapes. Keep non-BMP Unicode
    # literal (TOML disallows JSON surrogate pairs), and escape TOML's DEL.
    encoded_path = json.dumps(str(output_path), ensure_ascii=False).replace("\x7f", "\\u007f")
    source = source.replace(
        'path = "results/quickstart.json"', f'path = {encoded_path}'
    )
    config_path.write_text(source, encoding="utf-8")


def _write_phase_config(config_path, output_path):
    source = json.loads((ROOT / "configs" / "phase_tfim.json").read_text(encoding="utf-8"))
    source["evolution"]["dtau"] = 0.05
    source["evolution"]["beta_max"] = 0.2
    source["evolution"]["record_every_beta"] = 0.1
    source["evolution"]["trotter_order"] = 2
    source["truncation"]["epsilon"] = 1e-12
    source["truncation"]["max_bond"] = 32
    source["output"]["path"] = str(output_path)
    config_path.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")


def run_smoke(solve_binary, output_directory):
    solve_binary = Path(solve_binary).resolve(strict=True)
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=False)

    cases = (
        ("tfim", "quickstart.toml", "tfim.json", _write_tfim_config),
        ("phase-tfim", "phase_tfim.json", "phase-tfim.json", _write_phase_config),
    )
    for name, config_name, result_name, writer in cases:
        config_path = output_directory / config_name
        result_path = output_directory / result_name
        writer(config_path, result_path)
        subprocess.run([str(solve_binary), str(config_path)], cwd=ROOT, check=True)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        validate_result(result)
        print(f"{name}: {len(result['records'])} valid record(s) in {result_path}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit("usage: smoke.py SOLVE_BINARY NEW_OUTPUT_DIRECTORY")
    run_smoke(argv[0], argv[1])


if __name__ == "__main__":
    main()
