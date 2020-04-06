import sys

import setuptools
import versioneer


# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for people with outdated setuptools
# and/or pip.
min_version = (3, 6)
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

setuptools.setup(
    name='bluesky',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='danielballan',
    author_email=None,
    license="BSD (3-clause)",
    url="https://github.com/bluesky/bluesky",
    packages=setuptools.find_packages(),
    python_requires='>={}'.format('.'.join(str(n) for n in min_version)),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    entry_points={
        'console_scripts': [
            'bluesky-0MQ-proxy = bluesky.commandline.zmq_proxy:main',
        ]
    },
)
