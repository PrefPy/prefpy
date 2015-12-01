prefpy
======

Rank aggregation algorithms in the computer science field of computational social choice


Work In Progress
================

- This is the very initial version in a Python package form, further structural changes will be coming
- Module naming conventions will be changed; currently the algorithm files take the initials of the names of the papers from which they originate (e.g. "gmmra" for Generalized Method of Moments for Rank Aggregation)
- Mixture Model for Plackett-Luce EMM algorithm by Gormley & Murphy is forthcoming pending verification and testing of the method
- Generalized method of moments algorithm for Plackett-Luce
- Random utility model algorithms (verification of the implentation needs to be completed)


Installation
============

Install by running setup.py with Python 3.5 with the command

    python3 setup.py install

Symlink install while developing to keep changes in the code instead with the command

    python3 setup.py develop
