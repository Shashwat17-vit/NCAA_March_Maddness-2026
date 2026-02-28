"""
Setup script for NCAA March Madness prediction project
"""

from setuptools import setup, find_packages

setup(
    name='ncaa-march-madness',
    version='0.1.0',
    description='NCAA March Madness 2026 Prediction',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
    ],
)
