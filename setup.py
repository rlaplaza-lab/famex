from setuptools import setup, find_packages

setup(
    name="qme",
    version="0.1.0",
    description="Quick mechanistic exploration using MLP/NNPs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "pytest",
        "matplotlib",
    ],
    extras_require={
        "dev": [
            "pytest-cov",
            "black",
            "flake8",
        ],
        "ml": [
            "torch",
            "ase",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)