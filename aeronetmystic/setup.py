# -*- encoding: utf-8 -*-
from setuptools import setup, find_packages
from codecs import open
import os

basedir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(basedir, "README.md"), encoding="utf-8") as readmefile:
    long_description = readmefile.read()

setup(
    name="aeronetmystic",
    version="0.0.1",
    description="extract Aeronet Data for input into libRadtran simulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    author="James Barry",
    author_email="",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="",
    packages=find_packages(),    
    extra_require={
        "dev": ["check-manifest"],
        "test": [],
    },
    data_files=[],
    entry_points={
        "console_scripts": [
            "aeronet2mystic=aeronetmystic.pvcal_aerosol_input:main",
        ],
    },
    project_urls={
    },
)
