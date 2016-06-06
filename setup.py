from setuptools import setup

if __name__ == "__main__":
    setup(name="prefpy",
          version="0.8",
          description="Rank aggregation algorithms",
          classifiers=[
              "Development Status :: 3 - Alpha",
              "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
              "Programming Language :: Python :: 3.4",
              "Programming Language :: Python :: 3.5",
              "Intended Audience :: Science/Research",
              "Topic :: Scientific/Engineering"
          ],
          url="http://github.com/xialirong/prefpy",
          author="Peter Piech",
          license="GPL-3",
          packages=["prefpy"],
          zip_safe=False)
