from __future__ import (absolute_import, division, print_function)
import versioneer

try:
    from setuptools import setup
except ImportError:
    try:
        from setuptools.core import setup
    except ImportError:
        from distutils.core import setup

setup(
    name='bluesky',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='danielballan',
    author_email=None,
    license="BSD (3-clause)",
    url="https://github.com/danielballan/bluesky",
    packages=['bluesky', 'bluesky.tests', 'bluesky.testing'],
    package_data={'bluesky': ['schema/*.json']},
    requires=['jsonschema', 'traitlets', 'prettytable', 'cycler', 'history',
              'numpy', 'matplotlib'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.4",
    ],
)
