PrefPy
======

Rank aggregation algorithms in the computer science field of computational social choice


What's New
==========

- Experiments and tests have been factored out of the repository (now located at https://github.com/pdpiech/prefpy-experiments)
- Generalized method of moments algorithm for mixtures of Plackett-Luce models
- Implementation of EMM algorithm for mixtures of Plackett-Luce by Gormley & Murphy


Work In Progress
================

- This is an initial version of the Python package form, further structural changes will be coming
- Module naming conventions will be changed; currently the algorithm files take the initials of the names of the papers from which they originate (e.g. "gmmra" for Generalized Method of Moments for Rank Aggregation)
- Mixture Model for Plackett-Luce EMM algorithm by Gormley & Murphy is forthcoming pending verification and testing of the method
- Random utility model algorithms (verification of the implentation needs to be completed)


Installation
============

- Use of MATLAB optimization in this package requires Python 3.4 due to lack of support yet for Python 3.5 by the MATLAB Engine

Install by running setup.py with Python 3.4 (or greater) with the command

    python3 setup.py install

Symlink install while developing to keep changes in the code instead with the command

    python3 setup.py develop
