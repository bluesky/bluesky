import tomllib
from collections import defaultdict
from datetime import datetime, timedelta

import requests
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from packaging.version import parse as version_parse

# Define the release dates and core packages
py_releases = {
    "3.8": "Oct 14, 2019",
    "3.9": "Oct 5, 2020",
    "3.10": "Oct 4, 2021",
    "3.11": "Oct 24, 2022",
    "3.12": "Oct 2, 2023",
    "3.13": "Oct 1, 2024",
    "3.14": "Oct 1, 2025",
}

plus36 = timedelta(days=int(365 * 3))
plus24 = timedelta(days=int(365 * 2))
delta6month = timedelta(days=int(365 // 2))

# Calculate cutoff date
now = datetime.now()
cutoff = now - delta6month


def get_release_dates(package, support_time=plus24):
    """Fetch release and drop dates for package versions."""
    releases = {}
    print(f"Querying pypi.org for {package} versions...", end="", flush=True)
    response = requests.get(
        f"https://pypi.org/simple/{package}", headers={"Accept": "application/vnd.pypi.simple.v1+json"}
    ).json()
    print("OK")
    file_date = defaultdict(list)
    for f in response["files"]:
        ver = f["filename"].split("-")[1]
        version = Version(ver)
        if version.is_prerelease or version.micro != 0:
            continue
        release_date = None
        for format in ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"]:
            release_date = datetime.strptime(f["upload-time"], format)
        if release_date:
            file_date[version].append(release_date)
    release_date = {v: min(file_date[v]) for v in file_date}
    for ver, release_date in sorted(release_date.items()):
        drop_date = release_date + support_time
        if drop_date >= cutoff:
            releases[ver] = {"release_date": release_date, "drop_date": drop_date}
    return releases


# Gather release data for Python and packages
package_releases = {
    "python": {
        version: {
            "release_date": datetime.strptime(release_date, "%b %d, %Y"),
            "drop_date": datetime.strptime(release_date, "%b %d, %Y") + plus36,
        }
        for version, release_date in py_releases.items()
    }
}

# Get supported versions from pyproject.toml
with open("pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)
specifiers_set = SpecifierSet(pyproject["project"]["requires-python"])
supported_versions = [
    classifier.split("::")[-1].strip()
    for classifier in pyproject["project"]["classifiers"]
    if "Programming Language :: Python ::" in classifier
]
supported_versions = [v for v in supported_versions if version_parse(v) in specifiers_set]


# Check if any versions are outdated
output_body = ""
for package, versions in package_releases.items():
    for version, dates in versions.items():
        if dates["drop_date"] < now and version in supported_versions:
            output_body += f"{package} {version} is outdated as of {dates['drop_date'].strftime('%Y-%m-%d')}\n"
        elif dates["release_date"] < now and dates["drop_date"] > now and version not in supported_versions:
            output_body += (
                f"- {package} {version} is not yet supported but was released on "
                f"{dates['release_date'].strftime('%Y-%m-%d')}\n"
            )
if output_body:
    output_body = f"Outdated or unsupported Python versions. Consider addressing the following:\n{output_body}"

print(output_body, end="")
