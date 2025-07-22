import subprocess
from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--prefix",
        required=True,
        help="The prefix path for where to find and save data",
    )

    args = parser.parse_args()
    return args.prefix


def main():
    prefix = parse_args()

    for subdir in Path(prefix).iterdir():
        if subdir.name != "hex_55_100":
            continue
        for sample in subdir.iterdir():
            if sample.name in ["TENX111", "TENX114"]:
                continue
            print(f"Running for {sample.name}")
            subprocess.run(["./src/scripts/run-istar.sh", str(sample)])
            print(f"Run complete for {sample.name}")


if __name__ == "__main__":
    main()
