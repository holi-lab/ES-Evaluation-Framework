from pathlib import Path
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove the total_dialogue_turns_level line from jsonl files."
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="Directory that contains jsonl files to update.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_dir = args.target_dir
    target_line = "\\n- total_dialogue_turns_level: medium (6–8 turns)"

    for path in sorted(target_dir.glob("*.jsonl")):
        text = path.read_text(encoding="utf-8")
        new_text = text.replace(target_line, "")

        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            print(f"Updated {path}")
        else:
            print(f"No change needed for {path}")


if __name__ == "__main__":
    main()
