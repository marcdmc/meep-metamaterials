from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='meep_metamaterials',
    version='0.1.0',
    description='Metamaterial simulation with MEEP',
    long_description=readme,
    author='Marc de Miguel',
    author_email='marcdmc@mit.edu',
    url='https://github.com/kennethreitz/samplemod',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)