import argparse
from pathlib import Path

from spp_module import spp_solve


def existing_file(path: str) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        raise argparse.ArgumentTypeError(f"file not found: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SPP from RINEX observation (.o) and navigation (.n) files.")
    parser.add_argument("obs_file", type=existing_file, help="RINEX observation file, usually ending in .o")
    parser.add_argument("nav_file", type=existing_file, help="RINEX GPS navigation file, usually ending in .n")
    parser.add_argument("--pseudorange-code", default="C1", help="Pseudorange observation code to use, default: C1")
    parser.add_argument("--max-epochs", type=int, default=None, help="Maximum epochs to process, default: all")
    parser.add_argument("--output-json", default="spp_results.json", help="Output JSON path, default: spp_results.json")

    args = parser.parse_args()
    try:
        results = spp_solve(
            args.obs_file,
            args.nav_file,
            pseudorange_code=args.pseudorange_code,
            max_epochs=args.max_epochs,
            output_json=args.output_json,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Wrote {len(results)} epochs to {args.output_json}")


if __name__ == "__main__":
    main()
