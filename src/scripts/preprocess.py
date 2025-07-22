import os
from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    parser = ArgumentParser(exit_on_error=True)

    parser.add_argument(
        "-g",
        "--grid",
        choices=["hex", "sq"],
        required=True,
        help="The type of grid to use; either sparse hex or dense square",
    )
    parser.add_argument(
        "-d",
        "--diam",
        required=False,
        help="The diameter between spots. Required when grid type is `hex`",
    )
    parser.add_argument(
        "-D",
        "--dist",
        required=False,
        help="The distance between spots. Required when grid type is `hex`",
    )
    parser.add_argument(
        "-s",
        "--side",
        required=False,
        help="The side length for square spots. Required when grid type is `sq`",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        required=True,
        help="The prefix path for where to find and save data",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        required=False,
        help="The output directory",
        default=None,
    )

    args = parser.parse_args()

    if args.grid == "hex":
        d = args.diam
        dist = args.dist

        if not d or not dist:
            parser.print_usage()
            print(
                f"{Path(__file__).name}: error: the following arguments are required: -d/--diam, -D/--dist"
            )
            exit(1)

        return {
            "prefix": args.prefix,
            "out_dir": args.out_dir,
            "grid_type": "hex",
            "params": {"d": int(d), "dist": int(dist)},
        }

    if args.grid == "sq":
        s = args.side

        if not s:
            parser.print_usage()
            print(
                f"{Path(__file__).name}: error: the following arguments are required: -s/--side"
            )
            exit(1)

        return {
            "prefix": args.prefix,
            "out_dir": args.out_dir,
            "grid_type": "sq",
            "params": {"s": int(s)},
        }


def main():
    import cupy as cp

    args = parse_args()
    prefix = args["prefix"]

    if args["grid_type"] == "hex":
        from src.binning.tissue_positions_hex import run
        from src.binning.counts_hex import process_sample, join_cnts
    elif args["grid_type"] == "sq":
        from src.binning.tissue_positions_sq import run
        from src.binning.counts_sq import process_sample, join_cnts

    sample_ids = ["TENX111", "TENX114", "TENX147", "TENX148", "TENX149"]
    img_dir = "/opt/gpudata/sjne/HEST/data/wsis"

    excl_locs = set(["TENX114", "TENX147", "TENX148", "TENX149"])
    excl_cnts = set(["TENX114", "TENX147", "TENX148", "TENX149"])

    for sample_id in sample_ids:
        if not args["out_dir"]:
            out_dir = f"{prefix}/{sample_id}"
        else:
            out_dir = f"{args['out_dir']}/{sample_id}"

        os.makedirs(out_dir, exist_ok=True)

        if sample_id not in excl_locs:
            print(f"Processing {sample_id}...")
            img_path = f"{img_dir}/{sample_id}.tif"

            run(**args["params"], img_path=img_path, out_dir=out_dir)

            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

        if sample_id not in excl_cnts:
            process_sample(sample_id, out_dir)
            join_cnts(f"{out_dir}/cnts", out_dir)


if __name__ == "__main__":
    main()
