"""Fixed Cargo verification command plus pure output parsing helpers."""

import argparse
from dataclasses import dataclass
import json
import os
import re
import shlex
import subprocess
import sys
import time
import unicodedata
from pathlib import Path, PureWindowsPath
from typing import Optional


@dataclass(frozen=True)
class TestTotals:
    passed: int = 0
    failed: int = 0
    ignored: int = 0

    def __add__(self, other: "TestTotals") -> "TestTotals":
        if not isinstance(other, TestTotals):
            return NotImplemented
        return TestTotals(
            self.passed + other.passed,
            self.failed + other.failed,
            self.ignored + other.ignored,
        )


@dataclass(frozen=True)
class ParsedWarning:
    message: str
    lint: Optional[str]
    path: Optional[str]
    line: Optional[int]
    column: Optional[int]


@dataclass(frozen=True)
class ParsedCargoOutput:
    test_binaries: int
    tests: TestTotals
    warnings: tuple[ParsedWarning, ...]


class VerificationInternalError(Exception):
    """Raised when verification metadata cannot be safely interpreted."""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    output: str
    elapsed_seconds: float


PROFILES = {
    "routine": ("cargo", "test", "--all-targets", "--color", "never"),
}


@dataclass(frozen=True)
class KnownWarning:
    lint: Optional[str]
    message: str
    path: str


@dataclass(frozen=True)
class WarningClassification:
    known: tuple[ParsedWarning, ...]
    unknown_project: tuple[ParsedWarning, ...]
    external: tuple[ParsedWarning, ...]
    missing_known: tuple[KnownWarning, ...]


@dataclass(frozen=True)
class VerificationDecision:
    exit_code: int
    classification: WarningClassification


_ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_TEST_RESULT = re.compile(
    r"^test result:\s*.*?\b(?P<passed>\d+) passed;\s*"
    r"(?P<failed>\d+) failed;\s*(?P<ignored>\d+) ignored;",
    re.MULTILINE,
)
_LOCATION = re.compile(r"^\s*-->\s+(?P<path>.+):(?P<line>\d+):(?P<column>\d+)\s*$", re.MULTILINE)
_LINT_NOTE = re.compile(r"^\s*=\s*note:.*?#\[warn\(([^)]+)\)\]", re.MULTILINE)
_WARNING_SUMMARY = re.compile(
    r"^(?:.+ generated \d+ warning(?:s)?"
    r"(?: \((?:\d+ duplicate(?:s)?|run `cargo fix(?: [^`]*)?` to apply \d+ suggestion(?:s)?)\))?"
    r"|\d+ warning(?:s)? emitted)\.?$"
)
_UNSAFE_TEXT_CATEGORIES = {"Cc", "Cs", "Zl", "Zp"}


def _normalize_warning_message(message: str) -> str:
    return " ".join(message.split())


def strip_ansi(text: str) -> str:
    """Remove ANSI terminal control sequences from *text*."""
    return _ANSI.sub("", text)


def _relative_path(path: str, repo_root: Path) -> str:
    candidate = os.path.normpath(path)
    if not os.path.isabs(candidate):
        return Path(candidate).as_posix()
    root = os.path.normpath(os.path.abspath(os.fspath(repo_root)))
    try:
        if os.path.commonpath((root, candidate)) == root:
            return os.path.relpath(candidate, root).replace(os.sep, "/")
    except ValueError:
        pass
    return Path(candidate).as_posix()


def _warning_blocks(output: str):
    starts = [match for match in re.finditer(r"^warning:\s*", output, re.MULTILINE)]
    for index, start in enumerate(starts):
        next_warning = starts[index + 1].start() if index + 1 < len(starts) else len(output)
        boundary = re.search(
            r"^\s*(?:test result:|test\s+\S|running\s+|(?:Finished|Compiling|Checking|error:))",
            output[start.end():next_warning],
            re.MULTILINE,
        )
        end = start.end() + boundary.start() if boundary else next_warning
        block = output[start.end():end]
        yield block, boundary is not None


def _parse_warning(block: str, repo_root: Path) -> Optional[ParsedWarning]:
    lines = block.splitlines()
    headline = []
    for line in lines:
        if re.match(r"^\s*(?:-->|\||=|help:|note:)", line):
            break
        if line.strip():
            headline.append(line.strip())
    message = _normalize_warning_message(" ".join(headline))
    if not message:
        return None
    if _WARNING_SUMMARY.fullmatch(message):
        return None
    location = _LOCATION.search(block)
    lint_match = _LINT_NOTE.search(block)
    return ParsedWarning(
        message=message,
        lint=lint_match.group(1).strip() if lint_match else None,
        path=_relative_path(location.group("path"), repo_root) if location else None,
        line=int(location.group("line")) if location else None,
        column=int(location.group("column")) if location else None,
    )


def parse_cargo_output(output: str, repo_root: Path) -> ParsedCargoOutput:
    """Parse test summaries and structured compiler warnings without executing anything."""
    clean = strip_ansi(output)
    totals = TestTotals()
    for match in _TEST_RESULT.finditer(clean):
        totals += TestTotals(
            int(match.group("passed")), int(match.group("failed")), int(match.group("ignored"))
        )
    warnings = []
    inherited_lint = None
    for block, resets_lint in _warning_blocks(clean):
        parsed = _parse_warning(block, repo_root)
        if parsed is None:
            inherited_lint = None
            continue
        if parsed.lint is not None:
            inherited_lint = parsed.lint
        elif parsed.path is not None and not _is_absolute_path(parsed.path) and inherited_lint is not None:
            parsed = ParsedWarning(
                parsed.message, inherited_lint, parsed.path, parsed.line, parsed.column
            )
        else:
            inherited_lint = None
        warnings.append(parsed)
        if resets_lint:
            inherited_lint = None
    return ParsedCargoOutput(
        test_binaries=len(_TEST_RESULT.findall(clean)),
        tests=totals,
        warnings=tuple(warnings),
    )


def _validate_baseline_path(path: str) -> None:
    native = Path(path)
    windows = PureWindowsPath(path)
    if _is_absolute_path(path) or ".." in native.parts or ".." in windows.parts:
        raise VerificationInternalError("baseline warning path must be project-relative")


def _validate_baseline_text(value: str) -> None:
    if any(unicodedata.category(character) in _UNSAFE_TEXT_CATEGORIES for character in value):
        raise VerificationInternalError("known-warning values must contain single-line Unicode text")


def _is_absolute_path(path: str) -> bool:
    windows = PureWindowsPath(path)
    return os.path.isabs(path) or windows.is_absolute() or windows.root == "\\"


def load_known_warnings(path: Path) -> tuple[KnownWarning, ...]:
    """Load the strict, versioned known-warning baseline."""
    try:
        with path.open(encoding="utf-8") as baseline_file:
            payload = json.load(baseline_file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationInternalError("could not load known-warning baseline") from error
    if not isinstance(payload, dict) or type(payload.get("version")) is not int or payload["version"] != 1:
        raise VerificationInternalError("unsupported known-warning baseline version")
    warnings = payload.get("warnings")
    if not isinstance(warnings, list):
        raise VerificationInternalError("known-warning baseline must contain a warnings list")
    entries = []
    identities = set()
    for warning in warnings:
        if not isinstance(warning, dict) or set(warning) != {"lint", "message", "path"}:
            raise VerificationInternalError("known-warning entry has an invalid schema")
        lint = warning["lint"]
        message = warning["message"]
        warning_path = warning["path"]
        if lint is not None and not isinstance(lint, str):
            raise VerificationInternalError("known-warning lint must be a string or null")
        if not isinstance(message, str) or not isinstance(warning_path, str):
            raise VerificationInternalError("known-warning message and path must be strings")
        normalized_message = _normalize_warning_message(message)
        for value in (lint, warning_path):
            if value is not None:
                _validate_baseline_text(value)
        _validate_baseline_text(normalized_message)
        _validate_baseline_path(warning_path)
        identity = (lint, normalized_message, warning_path)
        if identity in identities:
            raise VerificationInternalError("duplicate known-warning identity")
        identities.add(identity)
        entries.append(KnownWarning(*identity))
    return tuple(entries)


def classify_warnings(
    warnings: tuple[ParsedWarning, ...], known: tuple[KnownWarning, ...]
) -> WarningClassification:
    """Classify parsed warnings by project ownership and baseline identity."""
    known_identities = {(warning.lint, warning.message, warning.path) for warning in known}
    seen_known = set()
    matches = []
    unknown_project = []
    external = []
    for warning in warnings:
        if warning.path is None or _is_absolute_path(warning.path):
            external.append(warning)
            continue
        identity = (warning.lint, warning.message, warning.path)
        if identity in known_identities:
            matches.append(warning)
            seen_known.add(identity)
        else:
            unknown_project.append(warning)
    missing = tuple(
        warning for warning in known if (warning.lint, warning.message, warning.path) not in seen_known
    )
    return WarningClassification(tuple(matches), tuple(unknown_project), tuple(external), missing)


def decide_verification(
    cargo_exit: int, parsed: ParsedCargoOutput, known: tuple[KnownWarning, ...]
) -> VerificationDecision:
    """Apply the wrapper's warning and parser exit policy."""
    classification = classify_warnings(parsed.warnings, known)
    if cargo_exit != 0:
        return VerificationDecision(cargo_exit, classification)
    if parsed.test_binaries == 0:
        return VerificationDecision(3, classification)
    if classification.unknown_project:
        return VerificationDecision(2, classification)
    return VerificationDecision(0, classification)


def _warning_groups(warnings: tuple[ParsedWarning, ...]) -> tuple[tuple[str, str, int], ...]:
    groups = {}
    for warning in warnings:
        key = (warning.lint or "unclassified", warning.path or "<unknown location>")
        groups[key] = groups.get(key, 0) + 1
    return tuple((lint, path, count) for (lint, path), count in sorted(groups.items()))


_SUMMARY_LINE_BYTES = 64
_MAX_KNOWN_GROUP_ROWS = 7
_MAX_EXTERNAL_GROUP_ROWS = 4
_MAX_MISSING_CANDIDATE_ROWS = 4


def _single_line_display(value: str) -> str:
    return "".join(
        "\ufffd" if unicodedata.category(character) == "Cs" else
        " " if unicodedata.category(character) in _UNSAFE_TEXT_CATEGORIES else
        character
        for character in value
    )


def _truncate_utf8(value: str, limit: int = _SUMMARY_LINE_BYTES) -> str:
    """Bound display text without splitting a UTF-8 sequence."""
    value = _single_line_display(value)
    if len(value.encode("utf-8")) <= limit:
        return value
    retained = []
    used = 0
    for character in value:
        encoded = character.encode("utf-8")
        if used + len(encoded) > limit - 3:
            break
        retained.append(character)
        used += len(encoded)
    return "".join(retained) + "..."


def _append_limited_rows(lines: list[str], rows: tuple[str, ...], limit: int, label: str) -> None:
    for row in rows[:limit]:
        lines.append(row)
    remaining = len(rows) - limit
    if remaining > 0:
        lines.append(f"  ... {remaining} additional {label} omitted")


def _display_command(command: tuple[str, ...]) -> str:
    """Return a bounded shell-parseable display form of *command*."""
    limit = _SUMMARY_LINE_BYTES - len("command: ".encode("utf-8"))
    displayed = []
    for argument in command:
        argument = _single_line_display(argument)
        candidate = shlex.join(tuple(displayed) + (argument,))
        if len(candidate.encode("utf-8")) <= limit:
            displayed.append(argument)
            continue
        abbreviated = ""
        for character in argument:
            shortened = abbreviated + character + "..."
            candidate = shlex.join(tuple(displayed) + (shortened,))
            if len(candidate.encode("utf-8")) > limit:
                break
            abbreviated += character
        if len(shlex.join(tuple(displayed) + (abbreviated + "...",)).encode("utf-8")) <= limit:
            displayed.append(abbreviated + "...")
        break
    return shlex.join(displayed)


def render_success_summary(
    profile: str,
    command: tuple[str, ...],
    parsed: ParsedCargoOutput,
    classification: WarningClassification,
    elapsed_seconds: float,
) -> str:
    """Render the compact, deterministic summary for a successful verification."""
    lines = [
        f"VERIFY {profile}: PASS",
        f"command: {_display_command(command)}",
        f"test binaries: {parsed.test_binaries}",
        (
            f"tests: {parsed.tests.passed} passed, {parsed.tests.failed} failed, "
            f"{parsed.tests.ignored} ignored"
        ),
        (
            f"warnings: {len(classification.known)} known, "
            f"{len(classification.unknown_project)} unknown-project, "
            f"{len(classification.external)} external"
        ),
        "known warning groups:",
    ]
    known_rows = tuple(
        f"  {lint}  {path}  {count}" for lint, path, count in _warning_groups(classification.known)
    )
    _append_limited_rows(lines, known_rows, _MAX_KNOWN_GROUP_ROWS, "known warning groups")
    if classification.external:
        lines.append("external warning groups:")
        external_rows = tuple(
            f"  {lint}  {path}  {count}"
            for lint, path, count in _warning_groups(classification.external)
        )
        _append_limited_rows(lines, external_rows, _MAX_EXTERNAL_GROUP_ROWS, "external warning groups")
    if classification.missing_known:
        lines.append("missing known-warning candidates:")
        missing_rows = tuple(
            f"  {warning.lint or 'unclassified'}  {warning.path}  {warning.message}"
            for warning in sorted(
            classification.missing_known,
            key=lambda item: (item.lint or "unclassified", item.path, item.message),
            )
        )
        _append_limited_rows(
            lines, missing_rows, _MAX_MISSING_CANDIDATE_ROWS, "known-warning candidates"
        )
    lines.append(f"elapsed: {elapsed_seconds:.1f}s")
    return "\n".join(_truncate_utf8(line) for line in lines) + "\n"


def run_command(command: tuple[str, ...], cwd: Path) -> CommandResult:
    """Run Cargo without a shell and return its complete combined output."""
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        output, _ = process.communicate()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    return CommandResult(process.returncode, output, time.monotonic() - started)


class _VerificationArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise VerificationInternalError(message)


def _profile_for_error(argv: list[str]) -> str:
    for argument in argv:
        if not argument.startswith("-"):
            return argument
    return "routine"


def _parser() -> argparse.ArgumentParser:
    parser = _VerificationArgumentParser(
        description=(
            "Run a fixed Cargo verification profile. The routine profile excludes doctests "
            "and ignored tests; the wrapper replays failed Cargo output completely."
        ),
        epilog=(
            "A nonzero Cargo status is preserved unchanged. A wrapper-generated exit 2 means "
            "an unknown project warning after successful Cargo execution. exit 0: verification "
            "passed. exit 3: invalid input, baseline, or unparseable successful output."
        ),
    )
    parser.add_argument("profile", nargs="?", metavar="profile", help="fixed profile: routine")
    return parser


def _render_policy_violation(profile: str, classification: WarningClassification) -> str:
    return (
        f"VERIFY {profile}: WARNING POLICY VIOLATION\n"
        f"known warnings: {len(classification.known)}\n"
        f"unknown-project warnings: {len(classification.unknown_project)}\n"
        f"external warnings: {len(classification.external)}\n"
    )


def main(argv: Optional[list[str]] = None) -> int:
    """Run one fixed verification profile and return its typed exit status."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    profile_for_error = _profile_for_error(arguments)
    try:
        try:
            namespace = _parser().parse_args(arguments)
        except SystemExit as error:
            if error.code == 0:
                return 0
            raise
        if namespace.profile is None:
            raise VerificationInternalError("a verification profile is required")
        profile = namespace.profile
        profile_for_error = profile
        if profile not in PROFILES:
            raise VerificationInternalError(f"unsupported verification profile: {profile}")

        root = Path(__file__).resolve().parents[1]
        result = run_command(PROFILES[profile], root)
        if result.returncode != 0:
            sys.stdout.write(result.output)
            return result.returncode

        parsed = parse_cargo_output(result.output, root)
        if parsed.test_binaries == 0:
            raise VerificationInternalError("cargo output contained no test results")
        known = load_known_warnings(Path(__file__).resolve().with_name("verify_known_warnings.json"))
        decision = decide_verification(result.returncode, parsed, known)
        if decision.exit_code == 2:
            sys.stdout.write(_render_policy_violation(profile, decision.classification))
            sys.stdout.write(result.output)
            return 2
        sys.stdout.write(
            render_success_summary(
                profile,
                PROFILES[profile],
                parsed,
                decision.classification,
                result.elapsed_seconds,
            )
        )
        return decision.exit_code
    except (VerificationInternalError, OSError, UnicodeError, json.JSONDecodeError) as error:
        sys.stderr.write(f"VERIFY {profile_for_error}: ERROR: {error}\n")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
