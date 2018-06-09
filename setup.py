#python setup.py sdist upload -r test
#python setup.py sdist upload

from setuptools import setup

# Extract version
def get_version():
    with open('surrkick/surrkick.py') as f:
        for line in f.readlines():
            if "__version__" in line:
                return line.split('"')[1]

def setup_package():

    metadata = dict(
        name='surrkick',
        version=get_version(),
        description='Black hole kicks from numerical-relativity surrogate models',
        long_description="See: `github.com/dgerosa/surrkick <https://github.com/dgerosa/surrkick>`_." ,
        classifiers=[
            'Topic :: Scientific/Engineering :: Astronomy',
            'Topic :: Scientific/Engineering :: Physics',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 2.7',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        keywords='gravitational-wave, black-hole binary',
        url='https://github.com/dgerosa/surrkick',
        author='Davide Gerosa',
        author_email='dgerosa@caltech.edu',
        license='MIT',
        packages=['surrkick'],
        install_requires=['numpy','scipy','matplotlib','singleton_decorator','tqdm','pathos','h5py','NRSur7dq2','precession'],
        include_package_data=True,
        zip_safe=False,
    )

    setup(**metadata)


if __name__ == '__main__':

    setup_package()
