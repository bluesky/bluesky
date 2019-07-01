import setuptools

import versioneer

with open('requirements.txt') as f:
    requirements = f.read().split()

setuptools.setup(
    name='bluesky',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='danielballan',
    author_email=None,
    license="BSD (3-clause)",
    url="https://github.com/NSLS-II/bluesky",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    entry_points={
        'console_scripts': [
            'bluesky-0MQ-proxy = bluesky.commandline.zmq_proxy:main',
        ]
    },
)
