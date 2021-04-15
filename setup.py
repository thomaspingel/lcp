from setuptools import setup, find_packages

setup(
    name='lcp',
    version='0.0.1',
    packages=['lcp',],
    license='MIT',
    long_description='Collection of tools for Least Cost Path Analysis in Python.',
    url='https://doi.org/10.1559/152304010791232163',

    author='Thomas Pingel',
    author_email='thomas.pingel@gmail.com',

    classifiers=['Development Status :: 4 - Beta',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.4',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Geographic Information Science',
                 'Intended Audience :: Science/Research',
                 'Operating System :: OS Independent',
                 'License :: OSI Approved :: MIT License'],
    keywords='GIS lidar least cost path LCP',
	install_requires=['scipy','pandas','python-igraph','numpy'],

	)