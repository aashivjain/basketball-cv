import argparse

from basketball_cv.config import load_player_tracking_config
from basketball_cv.tracking.players import track_players


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track basketball players in a video.")
    parser.add_argument(
        "--config",
        default="configs/player_tracking.yaml",
        help="Path to a player tracking YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_player_tracking_config(args.config)
    track_players(config)
    print("Player tracking complete.")


if __name__ == "__main__":
    main()
