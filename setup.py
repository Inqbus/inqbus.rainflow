from setuptools import setup, find_packages

version = '1.3'

setup(name='inqbus.rainflow',
      version=version,
      description="Cython optimized rainflow algorithm",
      long_description=open("README.rst").read() + "\n" +
                       open("HISTORY.txt").read()+ "\n" +
                       open("LICENSE.txt").read(),
      # Get more strings from
      # http://pypi.python.org/pypi?:action=list_classifiers
      classifiers=[
        "Programming Language :: Python",
        "Environment :: Plugins",
        "Intended Audience :: System Administrators",
        "License :: Other/Proprietary License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
                  ],
      keywords=['rainflow'],
      author='volker.jaenisch@inqbus.de',
      author_email='volker.jaenisch@inqbus.de',
      url='https://github.com/sandrarum/inqbus.rainflow',
      download_url='',
      license='',
      packages=find_packages('src', exclude=['']),
      namespace_packages=['inqbus'],
      package_dir={'': 'src'},
      include_package_data=True,
      zip_safe=False,
      extras_require=dict(
                           extra=[
                                  ],
                           docs=[
                                 'z3c.recipe.sphinxdoc',
                                 'sphinxcontrib-requirements',
                                 ],
                           test=[
                                'nose',
                                'coverage',
                                'unittest2'
                                ]
                           ),
      install_requires=[
          'setuptools',
          # -*- Extra requirements: -*-
          'numpy',
          'tables',
          'pandas',
          'cython',
          'numexpr',
      ],
      entry_points="""
      # -*- Entry points: -*-
      """
      )
