"""Make switcher.json to allow docs to switch between different versions."""

import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from subprocess import CalledProcessError, check_output


def report_output(stdout: bytes, label: str) -> list[str]:
    """Print and return something received frm stdout."""
    ret = stdout.decode().strip().split("\n")
    print(f"{label}: {ret}")
    return ret


def get_branch_contents(ref: str) -> list[str]:
    """Get the list of directories in a branch."""
    stdout = check_output(["git", "ls-tree", "-d", "--name-only", ref])
    return report_output(stdout, "Branch contents")


def get_sorted_tags_list() -> list[str]:
    """Get a list of sorted tags in descending order from the repository."""
    stdout = check_output(["git", "tag", "-l", "--sort=-v:refname"])
    return report_output(stdout, "Tags list")


def get_versions(ref: str, add: str | None) -> list[str]:
    """Generate the file containing the list of all GitHub Pages builds."""
    # Get the directories (i.e. builds) from the GitHub Pages branch
    try:
        builds = set(get_branch_contents(ref))
    except CalledProcessError:
        builds = set()
        logging.warning(f"Cannot get {ref} contents")

    # Add and remove from the list of builds
    if add:
        builds.add(add)

    # Get a sorted list of tags
    tags = get_sorted_tags_list()

    # Make the sorted versions list from main branches and tags
    versions: list[str] = []
    for version in ["master", "main"] + tags:
        if version in builds:
            versions.append(version)
            builds.remove(version)

    # Add in anything that is left to the bottom
    versions += sorted(builds)
    print(f"Sorted versions: {versions}")
    return versions


def write_json(path: Path, repository: str, versions: list[str]):
    """Write the JSON switcher to path."""
    org, repo_name = repository.split("/")
    struct = [
        {"version": version, "url": f"https://{org}.github.io/{repo_name}/{version}/"} for version in versions
    ]
    text = json.dumps(struct, indent=2)
    print(f"JSON switcher:\n{text}")
    path.write_text(text, encoding="utf-8")


def main(args=None):
    """Parse args and write switcher."""
    parser = ArgumentParser(description="Make a versions.json file from gh-pages directories")
    parser.add_argument(
        "--add",
        help="Add this directory to the list of existing directories",
    )
    parser.add_argument(
        "repository",
        help="The GitHub org and repository name: ORG/REPO",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path of write switcher.json to",
    )
    args = parser.parse_args(args)

    # Write the versions file
    versions = get_versions("origin/gh-pages", args.add)
    write_json(args.output, args.repository, versions)


if __name__ == "__main__":
    main()
