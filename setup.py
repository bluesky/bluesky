import sys

import setuptools
import versioneer


# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for people with outdated setuptools
# and/or pip.
min_version = (3, 8)
if sys.version_info < min_version:
    error = """
bluesky does not support Python {0}.{1}.
Python {2}.{3} and above is required. Check your Python version like so:

python3 --version

This may be due to an out-of-date pip. Make sure you have pip >= 9.0.1.
Upgrade pip like so:

pip install --upgrade pip
""".format(*(sys.version_info[:2] + min_version))
    sys.exit(error)

with open('requirements.txt') as f:
    requirements = f.read().split()
requirements.__add__(["event-model @ git+https://github.com/evalott100/event-model@98b9905df5b3c49c6bb96ecc32d8be3dc2ba41e2"])

with open('README.md', encoding='utf-8') as readme_file:
    readme = readme_file.read()

extras_require = {
    "ipython": ["ipython"],
    "zmq": ["pyzmq"],
    "common": ["ophyd", "databroker"],
    "tools": ["doct", "lmfit", "tifffile", "historydict"],
    "streamz": ["streamz"],
    "plotting": ["matplotlib"],
    "cmd": ["colorama"],
    "olog": ["jinja2"],
}

extras_require['all'] = sorted(set(sum(extras_require.values(), [])))

setuptools.setup(
    name='bluesky',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='danielballan',
    author_email=None,
    description="Experiment specification & orchestration.",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="BSD (3-clause)",
    url="https://github.com/bluesky/bluesky",
    packages=setuptools.find_packages(),
    package_data={"bluesky": ["py.typed"]},
    python_requires='>={}'.format('.'.join(str(n) for n in min_version)),
    install_requires=requirements,
    extras_require=extras_require,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    entry_points={
        'console_scripts': [
            'bluesky-0MQ-proxy = bluesky.commandline.zmq_proxy:main',
        ]
    },
)
