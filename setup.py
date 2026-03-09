from setuptools import setup, find_packages

setup(
    name="geometric-learning-dynamics",
    version="1.0.0",
    author="Mohsen Mostafa",
    author_email="mohsen.mostafa.ai@outlook.com",
    description="Geometric Learning Dynamics in Neural Networks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/[your-username]/geometric-learning-dynamics",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Mathematics :: Geometry",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "pyyaml>=5.4.0",
    ],
)
