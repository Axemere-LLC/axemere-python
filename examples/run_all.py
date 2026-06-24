"""Run all Axemere AI Gateway examples sequentially and report results.

Usage:
    cd examples/
    python run_all.py [--fail-fast] [--only 1,3] [--skip 10,11]

Options:
    --fail-fast     Stop after the first failure
    --only N[,N]    Run only these example numbers (e.g. --only 1,3 to re-run failures)
    --skip N[,N]    Skip these example numbers (ignored when --only is set)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


EXAMPLES_DIR = Path(__file__).parent

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
DIM = "\033[2m"


def colored(text: str, *codes: str) -> str:
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + RESET


def find_examples() -> list[tuple[int, Path]]:
    scripts = sorted(EXAMPLES_DIR.glob("[0-9][0-9]_*.py"))
    return [(int(p.stem[:2]), p) for p in scripts]


def check_skip(num: int, only: set[int], manual_skips: set[int]) -> str | None:
    """Return a skip reason string, or None if the example should run."""
    if only:
        return None if num in only else "not in --only list"
    if num in manual_skips:
        return "skipped via --skip"
    return None


def run_example(path: Path) -> tuple[bool, float, str]:
    """Run a single example script. Returns (passed, elapsed_seconds, output)."""
    start = time.monotonic()
    result = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        cwd=EXAMPLES_DIR,
    )
    elapsed = time.monotonic() - start
    output = result.stdout
    if result.stderr:
        output += result.stderr
    return result.returncode == 0, elapsed, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failure")
    parser.add_argument("--only", default="", help="Run only these comma-separated example numbers")
    parser.add_argument("--skip", default="", help="Comma-separated example numbers to skip (ignored when --only is set)")
    args = parser.parse_args()

    def parse_nums(val: str, flag: str) -> set[int]:
        result: set[int] = set()
        for part in val.split(","):
            part = part.strip()
            if part:
                try:
                    result.add(int(part))
                except ValueError:
                    print(f"Invalid {flag} value: {part!r}", file=sys.stderr)
                    sys.exit(1)
        return result

    only: set[int] = parse_nums(args.only, "--only")
    manual_skips: set[int] = parse_nums(args.skip, "--skip")

    # Load .env if python-dotenv is available
    try:
        from dotenv import load_dotenv
        env_path = EXAMPLES_DIR / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass

    examples = find_examples()
    if not examples:
        print("No example scripts found.", file=sys.stderr)
        sys.exit(1)

    results: list[tuple[int, Path, str, bool | None, float, str]] = []
    # (num, path, status_label, passed_or_None_if_skipped, elapsed, output)

    run_count = len(only) if only else len(examples) - len(manual_skips)
    print(colored(f"\nRunning {run_count}/{len(examples)} examples\n", BOLD))

    for num, path in examples:
        label = path.stem
        skip_reason = check_skip(num, only, manual_skips)

        if skip_reason:
            print(colored(f"  SKIP  ", YELLOW) + colored(label, BOLD) + f"  ({skip_reason})")
            results.append((num, path, "skip", None, 0.0, ""))
            continue

        print(colored(f"  RUN   ", CYAN) + colored(label, BOLD) + " ...", end="", flush=True)
        passed, elapsed, output = run_example(path)

        if passed:
            print(f"\r{colored('  PASS  ', GREEN)}{colored(label, BOLD)}  {colored(f'({elapsed:.1f}s)', DIM)}")
        else:
            print(f"\r{colored('  FAIL  ', RED)}{colored(label, BOLD)}  {colored(f'({elapsed:.1f}s)', DIM)}")

        results.append((num, path, "pass" if passed else "fail", passed, elapsed, output))

        if not passed:
            # Always show output for failures
            print(colored("  --- output ---", DIM))
            for line in output.rstrip().splitlines():
                print(f"    {line}")
            print(colored("  --- end ---", DIM))
            if args.fail_fast:
                print(colored("\nStopping (--fail-fast)", YELLOW))
                break

    # Summary
    passed_count = sum(1 for *_, p, _, _ in results if p is True)
    failed_count = sum(1 for *_, p, _, _ in results if p is False)
    skipped_count = sum(1 for *_, p, _, _ in results if p is None)
    total_run = passed_count + failed_count
    total_time = sum(elapsed for _, _, _, _, elapsed, _ in results)

    print()
    print(colored("Results", BOLD))
    print(f"  Passed:  {passed_count}/{total_run}")
    if skipped_count:
        print(f"  Skipped: {skipped_count}")
    if failed_count:
        print(colored(f"  Failed:  {failed_count}", RED))
        print()
        print(colored("Failed examples:", RED + BOLD))
        for num, path, status, passed, _, _ in results:
            if passed is False:
                print(f"  - {path.stem}")
    print(f"  Time:    {total_time:.1f}s")
    print()

    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
