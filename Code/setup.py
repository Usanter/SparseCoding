from __future__ import print_function
import sys
import os
from setuptools import setup, find_packages
with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]


print('Begin Installation')
try:
    import numpy
    print('Successfully import numpy')
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)
try:
    import tensorflow
    print('successfully import tensorflow')
except ImportError:
    print('Tensorflow is required during installation')
    sys.exit(1)
try:
    import scipy
    print('Successfully import scipy')
except ImportError:
    print('scipy is required during installation')
    sys.exit(1)

try:
    import sklearn
    print('Successfully import sklearn')
except ImportError:
    print('sklearn is required during installation')
    sys.exit(1)
try:
    import spams
    print('Successfully import spams')
except ImportError:
    print('spams is required during installation')
    sys.exit(1)

local_path = os.path.dirname(os.path.abspath(sys.argv[0]))

os.chdir(local_path)
sys.path.insert(0, local_path)


setup(name='DL-SF',
      version='0.0.1',
      description='A library for dictionary learning for extract new features in speech',
      long_description='A library for extract new features in speech '
                       'in an unsupervised, weekly supervised  and supervised scenario. ',
      url='https://github.com/Usanter/SparseCoding',
      author='Thomas Rolland',
      author_email='thomas.rolland@irit.fr',
      #license='boh',
      classifiers={
          'Development Status :: 0',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'Programming Language :: Python',
          'License :: OSI Approved :: BSD License',
          'Topic :: Software Development',
          'Topic :: Scientific/Engineering :: Signals processing',
          'Operating System :: POSIX',
          'Operating System :: Unix'},
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      #scripts=''
      )

print('\n \n Successfully install DL-SF')
