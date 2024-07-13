import re
from setuptools import setup, find_packages

VERSION = '0.0.1'

with open("README.md", "r") as readme:
    description = readme.read()

setup(
    name="viva-comets",
    version=VERSION,
    author="Amin Boroomand",
    author_email="abaroomand@gmail.com",
    description="Vivarium-COMETS is a multiscale modeling project that leverages the Vivarium package to implement the COMETS methodology, aiming to simulate and analyze microbial systems in spatial environments for understanding dynamic interactions.",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/vivarium-collective/VivaComets",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "vivarium-core>=1.6.4",
        "numpy",
        "cobra",
        "Pint==0.23",
        "matplotlib",
        "imageio[v2]",
        "ipython",
        "scipy",
        "pillow",
        "pandas",
        "logging",
        "warnings",
    ],
)
