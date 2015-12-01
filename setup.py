from setuptools import setup

if __name__ == "__main__":
    setup(name="prefpy",
          version="0.0.1",
          description="Rank aggregation algorithms",
          classifiers=[
              "Development Status :: 3 - Alpha",
              "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
              "Programming Language :: Python :: 3.5",
              "Intended Audience :: Science/Research",
              "Topic :: Scientific/Engineering"
          ],
          url="http://github.com/zmjjmz/prefpy",
          author="Peter Piech",
          license="GPL-3",
          packages=["rankpy"],
          zip_safe=False)
