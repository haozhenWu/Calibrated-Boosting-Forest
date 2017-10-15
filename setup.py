from setuptools import setup

setup(name='Calibrated-Boosting-Forest',
      version='0.1',
      description='Original implementation of Calibrated Boosting-Forest',
      url='https://github.com/haozhenWu/Calibrated-Boosting-Forest',
      author='Haozhen Wu',
      author_email='wuhaozhen@hotmail.com',
      license='GPL v3',
      packages=['lightchem'],
      setup_requires=['pytest-runner'],
      # Had Travis CI build problems when listing rdkit here, need to check versions
      # Can this switch to the official conda version?
      install_requires=['scikit-learn', 'xgboost', 'numpy', 'scipy', 'pandas'],
      tests_require=['pytest'],
      zip_safe=False)
