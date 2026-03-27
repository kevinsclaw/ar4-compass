from setuptools import setup, find_packages

setup(
    name="ar4-compass",
    version="0.1.0",
    author="Kevin Yang",
    description="Advanced Causal Discovery for Sim-to-Real Transfer on AR4 Robot",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kevinsclaw/ar4-compass",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "tensorboard>=2.11.0",
    ],
    extras_require={
        "mujoco": ["mujoco>=3.0.0", "gymnasium>=0.29.0"],
        "rl": ["stable-baselines3>=2.0.0"],
        "dev": ["pytest>=7.0.0", "black>=22.0.0", "flake8>=4.0.0"],
    },
)
