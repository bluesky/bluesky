from __future__ import (absolute_import, division, print_function)
import versioneer

import setuptools

setuptools.setup(
    name='bluesky',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='danielballan',
    author_email=None,
    license="BSD (3-clause)",
    url="https://github.com/NSLS-II/bluesky",
    packages=setuptools.find_packages(),
    package_data={'bluesky': ['schema/*.json']},
    install_requires=['jsonschema', 'traitlets', 'cycler',
                      'numpy', 'matplotlib', 'super_state_machine',
                      'historydict'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
    ],
)
