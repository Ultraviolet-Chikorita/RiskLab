"""
Setup script for Risk-Conditioned AI Evaluation Lab.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="risklab",
    version="0.1.0",
    author="AI Safety Lab",
    description="Risk-Conditioned AI Evaluation Lab - Measure and analyze manipulative behavior in AI systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "langgraph>=0.2.0",
        "langchain>=0.3.0",
        "langchain-openai>=0.2.0",
        "langchain-anthropic>=0.2.0",
        "openai>=1.0.0",
        "anthropic>=0.30.0",
        "transformers>=4.40.0",
        "huggingface-hub>=0.23.0",
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.0",
        "rich>=13.0.0",
        "typer>=0.12.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.0",
        ],
        "local": [
            "torch>=2.0.0",
            "accelerate>=0.30.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "risklab=risklab.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
