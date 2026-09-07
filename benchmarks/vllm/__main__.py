"""Run vLLM benchmark applications independently from qualification tests."""

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("application", choices=("models",))
    args, remaining = parser.parse_known_args(argv)
    if args.application == "models":
        from benchmarks.vllm.models.runner import main as run

        return run(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
