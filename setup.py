#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'bmcs_matmod'
DESCRIPTION = "Suite of utilities for material model development."
URL = 'https://github.com/bmcs-group/bmcs_matmod'
EMAIL = 'rostislav.chudoba@rwt-aachen.de'
AUTHOR = 'BMCS-Group'
REQUIRES_PYTHON = '>=3.6.0'

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Instead, read version from a file (e.g., bmcs_matmod/__version__.py)
about = {}
with open(os.path.join(here, 'bmcs_matmod', '__version__.py')) as f:
    exec(f.read(), about)
VERSION = about['__version__']


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        sys.exit()


class ReleaseCommand(Command):
    """Support setup.py upload."""

    description = 'Release the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


# Add these lines to define REQUIRED and EXTRAS before setup() is called:
REQUIRED = [
    # Add your runtime dependencies here, e.g.:
    'bmcs_utils',
    'numpy',
    'sympy',
    'traits',
]

EXTRAS = {
    'dev': [
        'pytest',
        'pytest-cov',
        'twine',
    ],
}


# Recommend using pyproject.toml for builds
if __name__ == "__main__":
    print(
        "Note: Modern Python packaging recommends using 'pyproject.toml' for builds and installs.\n"
        "You can still use this setup.py for legacy support, but 'pip install .' will use pyproject.toml if present."
    )
    # Where the magic happens:
    setup(
        name=NAME,
        version=about['__version__'],
        description=DESCRIPTION,
        long_description=long_description,
        long_description_content_type='text/markdown',
        author=AUTHOR,
        author_email=EMAIL,
        python_requires=REQUIRES_PYTHON,
        url=URL,
        packages=find_packages(exclude=('tests',)),
        # If your package is a single module, use this instead of 'packages':
        # py_modules=['mypackage'],

        # entry_points={
        #     'console_scripts': ['mycli=mymodule:cli'],
        # },
        install_requires=REQUIRED,
        extras_require=EXTRAS,
        include_package_data=True,
        license='MIT',
        classifiers=[
            # Trove classifiers
            # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: Implementation :: CPython',
            'Programming Language :: Python :: Implementation :: PyPy'
        ],
        # $ setup.py publish support.
        cmdclass={
            'upload': UploadCommand,
            'release': ReleaseCommand,
        },
    )