"""CLI entry point: python -m viewer --subject 157336 --profile debug"""

import argparse

from viewer.app import launch


def main():
    parser = argparse.ArgumentParser(description="Interactive volumetric viewer")
    parser.add_argument("--subject", default="157336", help="Subject ID")
    parser.add_argument("--profile", default="debug", help="Profile (debug/dev/prod)")
    args = parser.parse_args()
    launch(args.subject, args.profile)


if __name__ == "__main__":
    main()
