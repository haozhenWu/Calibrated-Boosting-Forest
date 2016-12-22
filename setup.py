from setuptools import setup

setup(name='lightchem',
      version='0.1',
      description='Light-weight Machine Learning models for Drug Discovery',
      url='https://github.com/haozhenWu/lightchem',
      author='Haozhen Wu',
      author_email='wuhaozhen@hotmail.com',
      license='GPL v3',
      packages=['lightchem'],
      setup_requires=['pytest-runner'],
      install_requires=['sklearn', 'xgboost', 'numpy', 'scipy', 'pandas', 'rdkit'],
      tests_require=['pytest'],
      zip_safe=False)
