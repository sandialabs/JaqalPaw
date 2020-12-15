"""Python tools for Jaqal"""

from setuptools import setup, find_packages

name = "JaqalPaw"
description = "Just Another Quantum Assembly Language Pulses and Waveform Specification"
version = "1.0.0rc1"

setup(
    name=name,
    description=description,
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    version=version,
    author="Daniel Lobser, Benjamin C. A. Morrison, Kenneth Rudinger, Antonio Russo, Jay Wesley Van Der Wall",
    author_email="qscout@sandia.gov",
    packages=find_packages(include=["jaqalpaw", "jaqalpaw.*"], where="src"),
    package_dir={"": "src"},
    install_requires=["numpy", "scipy", "jaqalpaq"],
    extras_require={
        "tests": ["pytest"],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "pygsti",
            f"jaqalpaq-extras[qiskit,pyquil,cirq,projectq,pytket]=={version}",
        ],
    },
    python_requires=">=3.6.5",
    platforms=["any"],
    url="https://qscout.sandia.gov",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Unix",
    ],
)
