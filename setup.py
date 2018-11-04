# -*- coding: utf-8 -*-

# Learn more: https://github.com/AnchorBlues/GroupLasso/blob/master/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='GroupLasso',
    version='0.1.0',
    description='Group Lasso package for Python',
    long_description=readme,
    author='Yu Umegaki',
    author_email='yu.umegaki@gmail.com',
    install_requires=['pandas', 'numpy', 'sklearn'],
    url='https://github.com/AnchorBlues/GroupLasso',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')), 
    test_suite='tests'
)
