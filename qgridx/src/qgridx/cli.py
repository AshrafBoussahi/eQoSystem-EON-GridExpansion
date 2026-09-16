"""Command-line entry point: ``qgridx <experiment> [options]``.

One command per reported result. Every experiment accepts ``--smoke``, which
runs the identical code path on a handful of instances at a reduced evaluation
budget; use it first to confirm the environment before committing hours of CPU.

Examples
--------
List what is available and where results will be written::

    qgridx --list

Confirm the install works, in about a minute::

    qgridx benchmark --smoke

Reproduce the main comparison table::

    qgridx benchmark --workers 16

Reproduce the security screening under data-center load growth::

    qgridx resilience --per-grid 10 --workers 60

Check the device path without submitting anything::

    qgridx hardware --stage validate
"""
import argparse
import importlib
import sys

EXPERIMENTS = {
    "benchmark": ("qgridx.experiments.benchmark",
                  "Main comparison table at a matched 10,000-evaluation budget"),
    "gate_budget": ("qgridx.experiments.gate_budget",
                    "Design sweep over the generator's gate allowance"),
    "scenarios": ("qgridx.experiments.scenarios",
                  "Thirteen-scenario stochastic formulation"),
    "resilience": ("qgridx.experiments.resilience",
                   "N-1 security screening under data-center load growth"),
    "microgrid": ("qgridx.experiments.microgrid",
                  "Storage sizing plus microgrid islanding in one encoding"),
    "resources": ("qgridx.experiments.resources",
                  "Measured resource table and planner-scale projection"),
    "hardware": ("qgridx.experiments.hardware",
                 "Device campaign (needs a provider; --stage selects a step)"),
}

HARDWARE_STAGES = ("prepare", "validate", "probe", "run", "collect", "analyze")


def build_parser():
    p = argparse.ArgumentParser(
        prog="qgridx",
        description="Qubit-efficient grid planning with correlation-encoded "
                    "optimization. Reproduces every reported result.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run `qgridx --list` to see every experiment and its output file.",
    )
    p.add_argument("experiment", nargs="?", choices=sorted(EXPERIMENTS),
                   help="which result to reproduce")
    p.add_argument("--list", action="store_true",
                   help="list experiments, then exit")
    p.add_argument("--smoke", action="store_true",
                   help="short run on a few instances, to check the environment")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel worker processes (default: all cores)")
    p.add_argument("--per-grid", type=int, default=None,
                   help="instances per grid, where the experiment sweeps grids")
    p.add_argument("--per-config", type=int, default=None,
                   help="instances per configuration, where applicable")
    p.add_argument("--shard", default=None, metavar="i/n",
                   help="run shard i of n, to split a sweep across machines")
    p.add_argument("--tag", default="", help="suffix for the output filename")
    p.add_argument("--stage", choices=HARDWARE_STAGES, default="validate",
                   help="hardware campaign stage (default: validate, which is "
                        "offline and free)")
    p.add_argument("--job-id", default=None,
                   help="hardware collect: the finished job to file")
    p.add_argument("--label", default=None, help="hardware collect: circuit label")
    p.add_argument("--axis", default=None, help="hardware collect: X, Y or Z")
    return p


def _print_list():
    from qgridx.utils.paths import results_dir
    print("qgridx experiments\n")
    for name, (_, desc) in sorted(EXPERIMENTS.items()):
        print(f"  {name:<12} {desc}")
    print(f"\nresults are written to: {results_dir()}")
    print("archived copies of every result ship with the package, so a fresh")
    print("run can be compared against the numbers in the paper.")


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.list or not args.experiment:
        _print_list()
        return 0

    module = importlib.import_module(EXPERIMENTS[args.experiment][0])

    if args.experiment == "hardware":
        stage = args.stage
        if stage == "collect":
            if not (args.label and args.axis and args.job_id):
                print("collect needs --label, --axis and --job-id", file=sys.stderr)
                return 2
            return module.collect(args.label, args.axis, args.job_id) or 0
        fn = getattr(module, stage)
        return fn() or 0

    kwargs = {"smoke": args.smoke}
    if args.workers is not None:
        kwargs["workers"] = args.workers
    if args.per_grid is not None:
        kwargs["per_grid"] = args.per_grid
    if args.per_config is not None:
        kwargs["per_cfg"] = args.per_config
    if args.shard:
        i, n = args.shard.split("/")
        kwargs["shard"] = (int(i), int(n))
    if args.tag:
        kwargs["tag"] = args.tag

    import inspect
    accepted = inspect.signature(module.main).parameters
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    return module.main(**kwargs) or 0


if __name__ == "__main__":
    raise SystemExit(main())
