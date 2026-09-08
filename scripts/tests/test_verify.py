from importlib.util import module_from_spec, spec_from_file_location
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from typing import Optional
import unittest
from unittest.mock import MagicMock, call, patch

SCRIPT = Path(__file__).resolve().parents[1] / "verify.py"
KNOWN_WARNINGS = SCRIPT.with_name("verify_known_warnings.json")
SPEC = spec_from_file_location("verify", SCRIPT)
verify = module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = verify
SPEC.loader.exec_module(verify)


class ParseCargoOutputTests(unittest.TestCase):
    def test_aggregates_all_test_result_rows(self):
        output = """
running 3 tests
test result: ok. 2 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out
running 4 tests
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
"""
        parsed = verify.parse_cargo_output(output, Path("/repo"))
        self.assertEqual(parsed.test_binaries, 2)
        self.assertEqual(parsed.tests, verify.TestTotals(6, 0, 1))

    def test_strips_ansi_before_parsing(self):
        output = "\x1b[32mtest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\x1b[0m\n"
        parsed = verify.parse_cargo_output(output, Path("/repo"))
        self.assertEqual(parsed.tests, verify.TestTotals(1, 0, 0))

    def test_parses_warning_location_and_lint_without_double_counting_summary(self):
        output = """
warning: function `cold_path` is never used
  --> src/example.rs:12:4
   |
   = note: `#[warn(dead_code)]` on by default
warning: crate generated 1 warning
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
"""
        parsed = verify.parse_cargo_output(output, Path("/repo"))
        self.assertEqual(parsed.warnings, (
            verify.ParsedWarning(
                message="function `cold_path` is never used",
                lint="dead_code",
                path="src/example.rs",
                line=12,
                column=4,
            ),
        ))

    def test_uses_only_compiler_note_as_warning_lint_source(self):
        output = """
warning: unknown lint: `obsolete`
  --> src/example.rs:1:8
   |
 1 | #[warn(obsolete)]
   |        ^^^^^^^^
   |
   = note: `#[warn(unknown_lints)]` on by default
"""
        parsed = verify.parse_cargo_output(output, Path("/repo"))
        self.assertEqual(parsed.warnings, (
            verify.ParsedWarning("unknown lint: `obsolete`", "unknown_lints", "src/example.rs", 1, 8),
        ))

    def test_normalizes_paths_and_handles_missing_location_and_lint(self):
        output = """
warning:   first line
    second line   
  --> /repo/src/lib.rs:3:8
   = note: `#[warn(unused)]` on by default
warning: external issue
  --> /outside/project.rs:9:2
warning: standalone issue
warning: 3 warnings emitted
"""
        parsed = verify.parse_cargo_output(output, Path("/repo"))
        self.assertEqual(parsed.warnings, (
            verify.ParsedWarning("first line second line", "unused", "src/lib.rs", 3, 8),
            verify.ParsedWarning("external issue", None, "/outside/project.rs", 9, 2),
            verify.ParsedWarning("standalone issue", None, None, None, None),
        ))

    def test_does_not_count_generated_warning_summary_with_duplicate_suffix(self):
        output = """
warning: function `cold_path` is never used
  --> src/example.rs:12:4
   = note: `#[warn(dead_code)]` on by default
warning: `example` (lib) generated 2 warnings (1 duplicate)
"""
        parsed = verify.parse_cargo_output(output, Path("/repo"))
        self.assertEqual(tuple(warning.message for warning in parsed.warnings), (
            "function `cold_path` is never used",
        ))

    def test_does_not_count_generated_warning_summary_with_cargo_fix_suffix(self):
        output = """
warning: unused import: `crate::cold_path`
  --> src/example.rs:3:5
   = note: `#[warn(unused_imports)]` on by default
warning: `example` (lib) generated 1 warning (run `cargo fix --lib -p example` to apply 1 suggestion)
"""
        parsed = verify.parse_cargo_output(output, Path("/repo"))
        self.assertEqual(tuple(warning.message for warning in parsed.warnings), (
            "unused import: `crate::cold_path`",
        ))

    def test_keeps_normal_diagnostic_that_only_resembles_generated_summary(self):
        output = """
warning: documentation generated 1 warning (during validation)
  --> src/example.rs:7:2
"""
        parsed = verify.parse_cargo_output(output, Path("/repo"))
        self.assertEqual(tuple(warning.message for warning in parsed.warnings), (
            "documentation generated 1 warning (during validation)",
        ))

    def test_stops_warning_at_test_and_indented_cargo_progress_boundaries(self):
        output = """
warning: first issue
test example::works ... ok
   Compiling example v0.1.0
warning: second issue
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.01s
"""
        parsed = verify.parse_cargo_output(output, Path("/repo"))
        self.assertEqual(parsed.warnings, (
            verify.ParsedWarning("first issue", None, None, None, None),
            verify.ParsedWarning("second issue", None, None, None, None),
        ))

    def test_inherits_lint_across_adjacent_rustc_warnings_when_cargo_omits_repeated_notes(self):
        output = """
warning: function `first` is never used
  --> src/example.rs:12:4
   |
   = note: `#[warn(dead_code)]` (part of `#[warn(unused)]`) on by default

warning: function `second` is never used
  --> src/example.rs:20:4

warning: field `third` is never read
  --> src/example.rs:24:5

warning: `example` (lib) generated 3 warnings
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.01s
"""
        parsed = verify.parse_cargo_output(output, Path("/repo"))
        self.assertEqual(
            tuple(warning.lint for warning in parsed.warnings),
            ("dead_code", "dead_code", "dead_code"),
        )

    def test_resets_inherited_lint_after_compiler_and_test_batch_boundaries(self):
        boundaries = (
            "    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.01s",
            "   Compiling next-crate v0.1.0",
            "test example::works ... ok",
            "running 1 test",
        )
        for boundary in boundaries:
            with self.subTest(boundary=boundary):
                output = f"""
warning: function `first` is never used
  --> src/first.rs:1:1
   = note: `#[warn(dead_code)]` on by default
{boundary}
warning: function `second` is never used
  --> src/second.rs:2:1
"""
                parsed = verify.parse_cargo_output(output, Path("/repo"))
                self.assertEqual(tuple(warning.lint for warning in parsed.warnings), ("dead_code", None))


class KnownWarningPolicyTests(unittest.TestCase):
    def write_baseline(self, value: object) -> Path:
        temporary = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self.addCleanup(lambda: Path(temporary.name).unlink(missing_ok=True))
        with temporary:
            json.dump(value, temporary)
        return Path(temporary.name)

    def test_load_known_warnings_normalizes_multiline_message_and_matches_parsed_warning(self):
        path = self.write_baseline({
            "version": 1,
            "warnings": [{
                "lint": "dead_code",
                "message": " function\n\t`cold_path`\x85is\u2028never\u2029used ",
                "path": "src/example.rs",
            }],
        })
        try:
            known = verify.load_known_warnings(path)
        except verify.VerificationInternalError as error:
            self.fail(f"multiline message should normalize before validation: {error}")
        self.assertEqual(known, (
            verify.KnownWarning("dead_code", "function `cold_path` is never used", "src/example.rs"),
        ))
        parsed_warning = verify.ParsedWarning(
            "function `cold_path` is never used", "dead_code", "src/example.rs", 12, 4
        )
        classification = verify.classify_warnings((parsed_warning,), known)
        self.assertEqual(classification.known, (parsed_warning,))
        self.assertEqual(classification.unknown_project, ())

    def test_load_known_warnings_rejects_invalid_schema(self):
        cases = (
            "{not json",
            {"version": 2, "warnings": []},
            {"version": True, "warnings": []},
            {"version": 1},
            {"version": 1, "warnings": {}},
            {"version": 1, "warnings": [{"lint": None, "message": "m"}]},
            {"version": 1, "warnings": [{"lint": None, "message": "m", "path": "x", "extra": 1}]},
            {"version": 1, "warnings": [{"lint": 3, "message": "m", "path": "x"}]},
            {"version": 1, "warnings": [{"lint": None, "message": 3, "path": "x"}]},
            {"version": 1, "warnings": [{"lint": None, "message": "m", "path": 3}]},
            {"version": 1, "warnings": [{"lint": None, "message": "m", "path": "/x"}]},
            {"version": 1, "warnings": [{"lint": None, "message": "m", "path": "../x"}]},
            {"version": 1, "warnings": [{"lint": None, "message": "m", "path": "C:\\x"}]},
            {"version": 1, "warnings": [{"lint": None, "message": "m", "path": "src\\..\\x"}]},
            {"version": 1, "warnings": [
                {"lint": None, "message": "m", "path": "x"},
                {"lint": None, "message": "m", "path": "x"},
            ]},
        )
        for case in cases:
            with self.subTest(case=case):
                if isinstance(case, str):
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temporary:
                        temporary.write(case)
                        path = Path(temporary.name)
                    self.addCleanup(lambda path=path: path.unlink(missing_ok=True))
                else:
                    path = self.write_baseline(case)
                with self.assertRaises(verify.VerificationInternalError):
                    verify.load_known_warnings(path)

    def test_load_known_warnings_rejects_invalid_utf8(self):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=False) as temporary:
            temporary.write(b'{"version": 1, "warnings": ["\xff"]}')
            path = Path(temporary.name)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        with self.assertRaises(verify.VerificationInternalError):
            verify.load_known_warnings(path)

    def test_load_known_warnings_rejects_unsafe_raw_lint_and_path_values(self):
        unsafe_values = (
            "line\nbreak",
            "tab\tvalue",
            "control\x85value",
            "line\u2028separator",
            "paragraph\u2029separator",
            "nul\x00value",
            "lone\ud800surrogate",
        )
        for field in ("lint", "path"):
            for unsafe in unsafe_values:
                with self.subTest(field=field, unsafe=ascii(unsafe)):
                    warning = {"lint": "dead_code", "message": "message", "path": "src/example.rs"}
                    warning[field] = unsafe
                    path = self.write_baseline({"version": 1, "warnings": [warning]})
                    with self.assertRaises(verify.VerificationInternalError):
                        verify.load_known_warnings(path)

    def test_load_known_warnings_rejects_surviving_unsafe_message_characters(self):
        for unsafe in ("nul\x00value", "bell\x07value", "lone\ud800surrogate"):
            with self.subTest(unsafe=ascii(unsafe)):
                path = self.write_baseline({
                    "version": 1,
                    "warnings": [{
                        "lint": "dead_code",
                        "message": unsafe,
                        "path": "src/example.rs",
                    }],
                })
                with self.assertRaises(verify.VerificationInternalError):
                    verify.load_known_warnings(path)

    def test_classification_ignores_line_movement_but_requires_lint_message_and_path(self):
        known = (verify.KnownWarning("dead_code", "function `cold_path` is never used", "src/example.rs"),)
        warnings = (
            verify.ParsedWarning("function `cold_path` is never used", "dead_code", "src/example.rs", 999, 1),
            verify.ParsedWarning("function `warm_path` is never used", "dead_code", "src/example.rs", 1, 1),
            verify.ParsedWarning("function `cold_path` is never used", "unused", "src/example.rs", 1, 1),
            verify.ParsedWarning("function `cold_path` is never used", "dead_code", "src/other.rs", 1, 1),
        )
        classification = verify.classify_warnings(warnings, known)
        self.assertEqual(classification.known, warnings[:1])
        self.assertEqual(classification.unknown_project, warnings[1:])
        self.assertEqual(classification.external, ())
        self.assertEqual(classification.missing_known, ())

    def test_classification_treats_portably_absolute_and_missing_paths_as_external(self):
        warnings = (
            verify.ParsedWarning("external", None, "/outside/example.rs", 1, 1),
            verify.ParsedWarning("windows external", None, "C:\\dep\\example.rs", 1, 1),
            verify.ParsedWarning("unc external", None, "\\\\server\\share\\example.rs", 1, 1),
            verify.ParsedWarning("rooted external", None, "\\outside\\example.rs", 1, 1),
            verify.ParsedWarning("unknown location", None, None, None, None),
        )
        classification = verify.classify_warnings(warnings, ())
        self.assertEqual(classification.external, warnings)
        self.assertEqual(classification.unknown_project, ())

    def test_classification_reports_missing_baseline_warning(self):
        known = (verify.KnownWarning("dead_code", "function `cold_path` is never used", "src/example.rs"),)
        classification = verify.classify_warnings((), known)
        self.assertEqual(classification.missing_known, known)


class VerificationDecisionTests(unittest.TestCase):
    def parsed(self, warnings=(), test_binaries=1):
        return verify.ParsedCargoOutput(test_binaries, verify.TestTotals(3, 0, 1), tuple(warnings))

    def test_decision_returns_two_for_unknown_project_warning(self):
        parsed = self.parsed((verify.ParsedWarning("new warning", None, "src/lib.rs", 1, 1),))
        decision = verify.decide_verification(0, parsed, ())
        self.assertEqual(decision.exit_code, 2)
        self.assertEqual(decision.classification.unknown_project, parsed.warnings)

    def test_decision_allows_known_external_and_missing_known_warnings(self):
        known = (verify.KnownWarning("dead_code", "known warning", "src/lib.rs"),)
        parsed = self.parsed((
            verify.ParsedWarning("known warning", "dead_code", "src/lib.rs", 1, 1),
            verify.ParsedWarning("dependency warning", None, "/dependency/src/lib.rs", 2, 1),
        ))
        self.assertEqual(verify.decide_verification(0, parsed, known).exit_code, 0)
        self.assertEqual(verify.decide_verification(0, self.parsed(), known).exit_code, 0)

    def test_decision_returns_three_when_success_has_no_test_result(self):
        parsed = verify.ParsedCargoOutput(0, verify.TestTotals(), ())
        self.assertEqual(verify.decide_verification(0, parsed, ()).exit_code, 3)

    def test_render_success_summary_is_compact_and_groups_deterministically(self):
        known = (
            verify.KnownWarning("dead_code", "a", "src/b.rs"),
            verify.KnownWarning("dead_code", "b", "src/a.rs"),
            verify.KnownWarning(None, "c", "src/a.rs"),
        )
        parsed = verify.ParsedCargoOutput(
            2,
            verify.TestTotals(7, 0, 1),
            (
                verify.ParsedWarning("a", "dead_code", "src/b.rs", 1, 1),
                verify.ParsedWarning("b", "dead_code", "src/a.rs", 2, 1),
                verify.ParsedWarning("c", None, "src/a.rs", 3, 1),
                verify.ParsedWarning("external", None, "/dependency/lib.rs", 4, 1),
            ),
        )
        classification = verify.classify_warnings(parsed.warnings, known)
        summary = verify.render_success_summary(
            "routine", ("cargo", "test", "space arg"), parsed, classification, 12.34
        )
        self.assertIn("VERIFY routine: PASS\n", summary)
        self.assertIn("command: cargo test 'space arg'\n", summary)
        self.assertIn("test binaries: 2\n", summary)
        self.assertIn("tests: 7 passed, 0 failed, 1 ignored\n", summary)
        self.assertIn("warnings: 3 known, 0 unknown-project, 1 external\n", summary)
        self.assertIn("known warning groups:\n  dead_code  src/a.rs  1\n  dead_code  src/b.rs  1\n  unclassified  src/a.rs  1\n", summary)
        self.assertIn("external warning groups:\n  unclassified  /dependency/lib.rs  1\n", summary)
        self.assertIn("elapsed: 12.3s\n", summary)
        self.assertNotIn("full captured cargo output", summary)
        self.assertLessEqual(len(summary.splitlines()), 30)
        self.assertLessEqual(len(summary.encode("utf-8")), 2048)

    def test_render_success_summary_includes_missing_known_candidates(self):
        known = (verify.KnownWarning("dead_code", "missing warning", "src/lib.rs"),)
        parsed = self.parsed()
        classification = verify.classify_warnings(parsed.warnings, known)
        summary = verify.render_success_summary("routine", ("cargo", "test"), parsed, classification, 0)
        self.assertIn("missing known-warning candidates:\n", summary)
        self.assertIn("  dead_code  src/lib.rs  missing warning\n", summary)

    def test_current_baseline_summary_stays_within_compact_output_budget(self):
        known = verify.load_known_warnings(KNOWN_WARNINGS)
        self.assertEqual(known, ())
        parsed = verify.ParsedCargoOutput(
            4,
            verify.TestTotals(120, 0, 0),
            tuple(
                verify.ParsedWarning(warning.message, warning.lint, warning.path, 1, 1)
                for warning in known
            ),
        )
        classification = verify.classify_warnings(parsed.warnings, known)
        summary = verify.render_success_summary(
            "routine", ("cargo", "test", "--all-targets", "--color", "never"),
            parsed, classification, 10.0,
        )
        self.assertLessEqual(len(summary.splitlines()), 30)
        self.assertLessEqual(len(summary.encode("utf-8")), 2048)

    def test_render_success_summary_bounds_many_long_groups_and_candidates(self):
        long = "⚠" * 1_000
        known = tuple(
            verify.KnownWarning(f"lint-{index}", long, f"src/{index}-{long}.rs")
            for index in range(20)
        )
        warnings = tuple(
            verify.ParsedWarning(item.message, item.lint, item.path, 1, 1) for item in known
        ) + tuple(
            verify.ParsedWarning(long, f"external-{index}", f"/dep/{index}-{long}.rs", 1, 1)
            for index in range(20)
        )
        parsed = verify.ParsedCargoOutput(1, verify.TestTotals(1, 0, 0), warnings)
        classification = verify.classify_warnings(warnings, known + tuple(
            verify.KnownWarning(f"missing-{index}", long, f"src/missing-{index}-{long}.rs")
            for index in range(20)
        ))
        summary = verify.render_success_summary(long, ("cargo", long), parsed, classification, 1.0)
        self.assertIn("VERIFY ", summary)
        self.assertIn("command: ", summary)
        self.assertIn("known warning groups:", summary)
        self.assertIn("external warning groups:", summary)
        self.assertIn("missing known-warning candidates:", summary)
        self.assertIn("additional known warning groups omitted", summary)
        self.assertIn("elapsed: 1.0s", summary)
        self.assertLessEqual(len(summary.splitlines()), 30)
        self.assertLessEqual(len(summary.encode("utf-8")), 2048)

    def test_render_success_summary_keeps_long_quoted_command_parseable(self):
        parsed = self.parsed()
        classification = verify.classify_warnings((), ())
        summary = verify.render_success_summary(
            "routine", ("cargo", "argument with spaces " + "x" * 1_000),
            parsed, classification, 1.0,
        )
        command = next(line.removeprefix("command: ") for line in summary.splitlines()
                       if line.startswith("command: "))
        self.assertEqual(shlex.split(command)[0], "cargo")
        self.assertLessEqual(len(summary.splitlines()), 30)
        self.assertLessEqual(len(summary.encode("utf-8")), 2048)

    def test_render_success_summary_sanitizes_display_fields_before_enforcing_bounds(self):
        known = tuple(
            verify.ParsedWarning(
                f"message\u2029forged-{index}",
                f"lint\x85forged-{index}",
                f"src/{index}\nforged.rs",
                1,
                1,
            )
            for index in range(20)
        )
        external = tuple(
            verify.ParsedWarning("external", "lint\rforged", f"/dep/{index}\u2028forged.rs", 1, 1)
            for index in range(20)
        )
        missing = tuple(
            verify.KnownWarning("dead_code", f"missing\nforged-{index}", f"src/missing-{index}.rs")
            for index in range(20)
        )
        parsed = verify.ParsedCargoOutput(1, verify.TestTotals(1, 0, 0), known + external)
        classification = verify.WarningClassification(known, (), external, missing)
        summary = verify.render_success_summary(
            "routine\nforged", ("cargo", "argument\u2028forged"), parsed, classification, 1.0
        )
        for unsafe in ("\r", "\x85", "\u2028", "\u2029"):
            self.assertNotIn(unsafe, summary)
        self.assertLessEqual(len(summary.splitlines()), 30)
        self.assertLessEqual(len(summary.encode("utf-8")), 2048)

    def test_render_success_summary_replaces_lone_surrogates(self):
        warning = verify.ParsedWarning("message", "lint\ud800", "src/example.rs", 1, 1)
        parsed = verify.ParsedCargoOutput(1, verify.TestTotals(1, 0, 0), (warning,))
        classification = verify.WarningClassification((warning,), (), (), ())
        summary = verify.render_success_summary("routine", ("cargo",), parsed, classification, 1.0)
        self.assertNotIn("\ud800", summary)
        self.assertLessEqual(len(summary.encode("utf-8")), 2048)


class RunCommandInterruptTests(unittest.TestCase):
    def test_keyboard_interrupt_terminates_waits_and_reraises(self):
        process = MagicMock()
        process.communicate.side_effect = KeyboardInterrupt()
        process.wait.return_value = 0
        with patch.object(verify.subprocess, "Popen", return_value=process) as popen:
            with self.assertRaises(KeyboardInterrupt):
                verify.run_command(("cargo", "test"), Path("/repo"))
        popen.assert_called_once_with(
            ("cargo", "test"),
            cwd=Path("/repo"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.assertEqual(
            process.method_calls,
            [call.communicate(), call.terminate(), call.wait(timeout=1)],
        )

    def test_keyboard_interrupt_kills_after_terminate_timeout_and_reraises(self):
        process = MagicMock()
        process.communicate.side_effect = KeyboardInterrupt()
        process.wait.side_effect = (subprocess.TimeoutExpired("cargo", 1), 0)
        with patch.object(verify.subprocess, "Popen", return_value=process):
            with self.assertRaises(KeyboardInterrupt):
                verify.run_command(("cargo", "test"), Path("/repo"))
        self.assertEqual(
            process.method_calls,
            [
                call.communicate(),
                call.terminate(),
                call.wait(timeout=1),
                call.kill(),
                call.wait(),
            ],
        )


class RoutineCliTests(unittest.TestCase):
    def run_main(self, argv, result=None):
        stdout = StringIO()
        stderr = StringIO()
        runner = None
        if result is not None:
            runner = patch.object(verify, "run_command", return_value=result)
        with (runner or _NoopContext()), redirect_stdout(stdout), redirect_stderr(stderr):
            code = verify.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def successful_output(self):
        return "test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n"

    def test_routine_uses_fixed_command_and_script_relative_repository_root(self):
        result = verify.CommandResult(0, self.successful_output(), 1.25)
        with patch.object(verify, "run_command", return_value=result) as run_command:
            code, stdout, stderr = self.run_main(["routine"])
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertIn("VERIFY routine: PASS\n", stdout)
        run_command.assert_called_once_with(
            ("cargo", "test", "--all-targets", "--color", "never"),
            Path(verify.__file__).resolve().parents[1],
        )

    def test_cargo_success_prints_only_compact_summary(self):
        cargo_output = self.successful_output() + "verbose cargo output\n"
        code, stdout, stderr = self.run_main(
            ["routine"], verify.CommandResult(0, cargo_output, 1.25)
        )
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertIn("VERIFY routine: PASS\n", stdout)
        self.assertNotIn(cargo_output, stdout)

    def test_cargo_failure_replays_complete_combined_output_and_preserves_exit_code(self):
        cargo_output = "warning: cargo warning\nerror: test failure\n"
        for cargo_exit in (1, 2, 101):
            with self.subTest(cargo_exit=cargo_exit):
                code, stdout, stderr = self.run_main(
                    ["routine"], verify.CommandResult(cargo_exit, cargo_output, 0.1)
                )
                self.assertEqual(code, cargo_exit)
                self.assertEqual(stdout, cargo_output)
                self.assertEqual(stderr, "")

    def test_unknown_project_warning_reports_classification_and_complete_output(self):
        cargo_output = (
            "warning: fresh warning\n"
            "  --> src/new.rs:1:1\n"
            "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n"
        )
        code, stdout, stderr = self.run_main(
            ["routine"], verify.CommandResult(0, cargo_output, 0.1)
        )
        self.assertEqual(code, 2)
        self.assertEqual(stderr, "")
        self.assertIn("VERIFY routine: WARNING POLICY VIOLATION\n", stdout)
        self.assertIn("unknown-project warnings: 1\n", stdout)
        self.assertTrue(stdout.endswith(cargo_output))
        self.assertEqual(stdout.count(cargo_output), 1)

    def test_invalid_baseline_returns_internal_error(self):
        stdout = StringIO()
        stderr = StringIO()
        with patch.object(
            verify,
            "load_known_warnings",
            side_effect=verify.VerificationInternalError("bad baseline"),
        ), patch.object(
            verify,
            "run_command",
            return_value=verify.CommandResult(0, self.successful_output(), 0.1),
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            code = verify.main(["routine"])
        self.assertEqual(code, 3)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "VERIFY routine: ERROR: bad baseline\n")

    def test_unicode_error_returns_internal_error_without_traceback(self):
        unicode_error = UnicodeEncodeError("utf-8", "\ud800", 0, 1, "surrogates not allowed")
        with patch.object(verify, "render_success_summary", side_effect=unicode_error):
            code, stdout, stderr = self.run_main(
                ["routine"], verify.CommandResult(0, self.successful_output(), 0.1)
            )
        self.assertEqual(code, 3)
        self.assertEqual(stdout, "")
        self.assertTrue(stderr.startswith("VERIFY routine: ERROR: "))
        self.assertNotIn("Traceback", stderr)

    def test_cargo_failure_precedes_baseline_loading(self):
        cargo_output = "error: test failure\n"
        stdout = StringIO()
        stderr = StringIO()
        with patch.object(
            verify,
            "load_known_warnings",
            side_effect=verify.VerificationInternalError("bad baseline"),
        ), patch.object(
            verify,
            "run_command",
            return_value=verify.CommandResult(1, cargo_output, 0.1),
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            code = verify.main(["routine"])
        self.assertEqual(code, 1)
        self.assertEqual(stdout.getvalue(), cargo_output)
        self.assertEqual(stderr.getvalue(), "")

    def test_successful_unparseable_output_returns_internal_error(self):
        code, stdout, stderr = self.run_main(
            ["routine"], verify.CommandResult(0, "cargo did not run tests\n", 0.1)
        )
        self.assertEqual(code, 3)
        self.assertEqual(stdout, "")
        self.assertEqual(stderr, "VERIFY routine: ERROR: cargo output contained no test results\n")

    def test_script_help_documents_fixed_profile_exclusions_failure_replay_and_exits(self):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("routine", completed.stdout)
        self.assertIn("excludes", completed.stdout)
        self.assertIn("replays", completed.stdout)
        self.assertIn("nonzero Cargo status is preserved", completed.stdout)
        self.assertIn("wrapper-generated exit 2", completed.stdout)
        self.assertIn("exit 2", completed.stdout)
        self.assertIn("exit 3", completed.stdout)

    def test_script_missing_profile_uses_internal_error_exit(self):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(completed.returncode, 3)
        self.assertIn("VERIFY routine: ERROR:", completed.stdout)

    def test_script_unsupported_profile_uses_internal_error_exit(self):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "unsupported"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(completed.returncode, 3)
        self.assertIn("VERIFY unsupported: ERROR:", completed.stdout)


class ContributingGuidanceTests(unittest.TestCase):
    def test_contributing_routes_routine_verification_without_narrowing_scope(self):
        contributing = (SCRIPT.parents[1] / "CONTRIBUTING.md").read_text(
            encoding="utf-8"
        )
        for required in [
            "python3 scripts/verify.py routine",
            "cargo test --doc",
            "ignored release benchmarks",
            "does not reduce verification scope",
        ]:
            self.assertIn(required, contributing)


class _NoopContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


if __name__ == "__main__":
    unittest.main()
