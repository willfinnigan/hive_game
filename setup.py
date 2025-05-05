from setuptools import setup, find_packages

setup(
    name="hive",
    version="0.1.0",
    description="Hive board game implementation",
    author="Will Finnigan",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[],
    extras_require={
        "dev": [
            "pytest",
        ],
    },
)