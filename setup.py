from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='libics',
      version='0.0',
      description='Incoherent source library',
      long_description=readme(),
      classifiers=[
            'Development Status :: 2 - Pre-Alpha',
            'Programming Language :: Python :: 3',
      ],
      keywords='data analysis incoherent light',
      url='https://github.com/david-wei/libics',
      author='David Wei',
      author_email='david94.wei@gmail.com',
      license='',
      packages=find_packages(),
      install_requires=[
            'PyQt5',
            'matplotlib', 'numpy', 'scipy',
            'pyserial'
      ],
      dependency_links=['https://github.com/morefigs/pymba'],
      include_package_data=True,
      zip_safe=False)
