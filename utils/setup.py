from setuptools import setup

setup( name='nesting_app_utils', 
      version='0.1', 
      description='A sample Python package', 
      packages=['bvh', 'polygonization', 'sdf'], 
      install_requires=[ 'numpy', 'pandas', ], )

