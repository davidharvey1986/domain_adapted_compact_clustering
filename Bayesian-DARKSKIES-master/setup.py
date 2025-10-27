"""
Package setup
"""
from setuptools import setup

# pylint: disable=unused-import
import src

setup(
    name='darkskies',
    version='0.0.1',
    description='DARKSKIES SIDM Clustering',
    url='https://github.com/EthanTreg/Bayesian-DARKSKIES',
    author='Ethan Tregidga',
    author_email='ethan.tregidga@epfl.ch',
    license='MIT',
    packages=['src', 'src.plots', 'src.utils'],
)
