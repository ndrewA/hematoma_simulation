"""CLI entry point: python -m viewer --subject 157336 --profile debug"""

import argparse

from viewer.app import launch


def main():
    parser = argparse.ArgumentParser(description="Interactive volumetric viewer")
    parser.add_argument("--subject", default="157336", help="Subject ID")
    parser.add_argument("--profile", default="debug", help="Profile (debug/dev/prod)")
    parser.add_argument("--capture", nargs="?", const="slices",
                        choices=["slices", "3d"],
                        help="Render one frame and save as viewer_screenshot.png (default: slices)")
    args = parser.parse_args()
    launch(args.subject, args.profile, capture=args.capture)


if __name__ == "__main__":
    main()
