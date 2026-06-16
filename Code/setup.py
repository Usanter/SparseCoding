from pathlib import Path

from setuptools import setup


BASE_DIR = Path(__file__).resolve().parent
README_PATH = BASE_DIR.parent / "README.md"
REQUIREMENTS_PATH = BASE_DIR / "requirements.txt"

INSTALL_REQUIRES = [
    line.strip()
    for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.strip().startswith("#")
]

setup(
    name="sparsecoding",
    version="0.1.0",
    description="Sparse coding utilities and a reproducible digits demo.",
    long_description=README_PATH.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/Usanter/SparseCoding",
    author="Thomas Rolland",
    py_modules=["SparseCoding", "demo"],
    install_requires=INSTALL_REQUIRES,
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
)
