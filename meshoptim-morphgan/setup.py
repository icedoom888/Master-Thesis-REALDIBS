#! /usr/bin/env python
"""A template for classifying images."""

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('noice', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'noice-toolbox'
DESCRIPTION = (
    'A toolbox for machine learning with noise'
    ' used at Disney Research|Studios.'
)
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Hayko Riemenschneider, Joan Massich'
MAINTAINER_EMAIL = 'hayko@disneyresearch.com, joan.massich@disneyresearch.com'
URL = 'https://bitbucket.org/disneyresearch/noicetoolbox/'
DOWNLOAD_URL = (
    'https://bitbucket.org/disneyresearch/noicetoolbox/downloads/'
)
VERSION = __version__  # noqa
INSTALL_REQUIRES = ['numpy']
CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'Programming Language :: Python',
    'Topic :: Software Development',
    'Topic :: Scientific/Engineering',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS',
    'Programming Language :: Python :: 3.7',
]
EXTRAS_REQUIRE = {
    'tests': ['pytest', 'pytest-cov'],
    'docs': ['sphinx', 'sphinx-gallery', 'sphinx_rtd_theme', 'numpydoc', 'matplotlib'],
}

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    zip_safe=False,  # the package can run out of an .egg file
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
