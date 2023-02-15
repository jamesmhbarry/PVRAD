# -*- encoding: utf-8 -*-
from setuptools import setup, find_packages
from codecs import open
import os

basedir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(basedir, "README.md"), encoding="utf-8") as readmefile:
    long_description = readmefile.read()

setup(
    name="cosmopvcal",
    version="0.0.1",
    description="extract surface properties from cosmo file for use in pv model calibration",
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
    install_requires=["pyresample", "xarray"],
    extra_require={
        "dev": ["check-manifest"],
        "test": [],
    },
    data_files=[],
    entry_points={
        "console_scripts": [
            "cosmo2pvcal=cosmopvcal.cosmo_surface_props:main",
        ],
    },
    project_urls={
    },
)
