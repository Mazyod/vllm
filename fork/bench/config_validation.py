# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validate committed engine YAML with the installed release's serve parser."""

import argparse

from fork.bench.profiles import ProfileStore, load


def validate_store(store: ProfileStore) -> str:
    """Parse every engine file with this installation's real serve parser.

    Args:
        store: Tag-selected gate and manual configurations.

    Returns:
        Installed vLLM version.

    Raises:
        RuntimeError: If the installed version does not match the store tag.
        SystemExit: If argparse rejects any engine key or value.
    """
    from vllm import __version__
    from vllm.entrypoints.cli.serve import ServeSubcommand
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    expected = store.tag.removeprefix("v")
    if __version__ != expected:
        raise RuntimeError(
            f"config {store.tag} requires vLLM {expected}, installed {__version__}"
        )
    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser")
    ServeSubcommand().subparser_init(subparsers)
    for engine_path in store.engine_paths():
        parser.parse_args(["serve", "--config", str(engine_path)])
    return __version__


def main(argv: list[str] | None = None) -> int:
    """Validate a release selected on the command line."""
    parser = argparse.ArgumentParser(prog="fork.bench.config_validation")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args(argv)
    print(validate_store(load(args.tag)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
