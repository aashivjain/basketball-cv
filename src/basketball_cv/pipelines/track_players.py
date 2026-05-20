import argparse
import logging
from pathlib import Path

from basketball_cv.config import load_player_tracking_config, PlayerTrackingConfig
from basketball_cv.tracking.players import track_players


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track basketball players in a video.")
    parser.add_argument(
        "--config",
        default="configs/player_tracking.yaml",
        help="Path to a player tracking YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional override for the output directory.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    config = load_player_tracking_config(args.config)

    if args.output_dir is not None:
        config.output_dir = args.output_dir

    config.validate()
    track_players(config)
    logging.info("Player tracking complete.")
    logging.info("Output directory: %s", config.output_dir)


if __name__ == "__main__":
    main()
