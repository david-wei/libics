from setuptools import setup, find_packages
__version__ = "1.1a"


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='libics',
      version=__version__,
      description='Incoherent source library',
      long_description=readme(),
      classifiers=[
            'Development Status :: 2 - Pre-Alpha',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Programming Language :: Python :: 3',
      ],
      keywords='data analysis driver',
      url='https://github.com/david-wei/libics',
      author='David Wei',
      author_email='david94.wei@gmail.com',
      license='GPLv3',
      packages=find_packages(),
      entry_points={
            'console_scripts': [
                  # 'script_name = package.module:function'
            ],
            'gui_scripts': []
      },
      install_requires=[
            'matplotlib', 'numpy', 'pandas', 'Pillow',
            'scipy', 'xxhash', 'colorama', 'colorspacious'
      ],
      extras_require={
            'bson': ['pymongo'], 'db': ['pika', 'influxdb-client']
      },
      python_requires='>=3.6',
      dependency_links=[
            'https://github.com/fujii-team/sif_reader'
      ],
      include_package_data=True,
      zip_safe=False)
