from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='Similarity Forest',
      version='0.1',
      description='Similarity Forest',
      packages=['simforest'],
      install_requires=['numpy'],
      scripts=[],
      zip_safe=False)
