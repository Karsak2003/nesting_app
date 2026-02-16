from setuptools import setup

setup( name='nesting_app_CORE', 
      version='0.1', 
      description='A sample Python package', 
      packages=['agent', 'collision', 'constraints', 'dynamics', 'geometry', 'optimizer'], 
      install_requires=[ 'numpy', 'pandas', 'utils'], )

