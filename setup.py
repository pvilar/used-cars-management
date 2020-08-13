""" Setup script for package """

from setuptools import setup, find_packages

with open("requirements.txt") as reqs:
    requirements = reqs.read().splitlines()

setup(
    name="carnext",
    version="0.1",
    description='Predictive model for Carnext data science case',
    author='Pau Vilar',
    author_email='pau.vilar.ribo@gmail.com',
    packages=find_packages(),
    python_requires=">=3.5",
    install_requires=requirements
)
