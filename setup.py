from setuptools import setup, find_packages

setup(
    name="rankpool",
    version="1.0.0",
    author="XiaoxuStu",
    description="RankPool: Rank-Weighted Pooling for Robust Feature Aggregation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/XiaoxuStu/RankPool",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "numpy>=1.19.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
