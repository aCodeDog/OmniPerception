from setuptools import setup, find_packages

setup(name='LidarSensor',
      version='1.0.0',
      author='Nikita Rudin',
      author_email='rudinn@ethz.ch',
      license="BSD-3-Clause",
      packages=find_packages(),
      description='Fast and simple RL algorithms implemented in pytorch',
      python_requires='>=3.8',
      install_requires=[
            "torch>=1.4.0",
            "numpy>=1.16.4"
      ],
      )
