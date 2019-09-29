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
      entry_points={
            'console_scripts': [
                  # 'script_name = package.module:function'
            ],
            'gui_scripts': []
      },
      install_requires=[
            'matplotlib', 'numpy', 'scipy', 'pandas',
            'pyserial', 'pymongo', 'h5py', 'Pillow', 'PyQt5'
      ],
      dependency_links=[
            'https://github.com/morefigs/pymba',
            'https://github.com/fujii-team/sif_reader'
      ],
      include_package_data=True,
      zip_safe=False)
