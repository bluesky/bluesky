from __future__ import (absolute_import, division, print_function)

try:
    from setuptools import setup
except ImportError:
    try:
        from setuptools.core import setup
    except ImportError:
        from distutils.core import setup

setup(
    name='bluesky',
    version='0.0.0.post1',
    author='danielballan',
    author_email=None,
    license="BSD (3-clause)",
    url="https://github.com/danielballan/bluesky",
    packages=['bluesky', 'bluesky.tests', 'bluesky.testing'],
    package_data={'bluesky': ['schema/*.json']},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.4",
    ],
)
