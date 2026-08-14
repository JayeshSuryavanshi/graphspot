from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="graphspot")
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("bench", help="reproduce the benchmark table from one command")
    bench.add_argument(
        "--quick",
        action="store_true",
        help="the four auto-download datasets, three seeds (yelpchi amazon tolokers questions)",
    )
    bench.add_argument("--datasets", nargs="*", default=None)
    bench.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    bench.add_argument("--regime", choices=["supervised", "semi"], default="supervised")

    args = parser.parse_args(argv)
    if args.command == "bench":
        from graphspot.bench import format_table, run_bench
        from graphspot.datasets import QUICK_LOADERS

        names = args.datasets if args.datasets else sorted(QUICK_LOADERS)
        if args.quick and not args.datasets:
            names = ["yelpchi", "amazon", "tolokers", "questions"]
        rows = run_bench(names, seeds=tuple(args.seeds), regime=args.regime)
        print(format_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
