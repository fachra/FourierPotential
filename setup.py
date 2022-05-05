from setuptools import setup

version = {}
with open('fpm/_version.py') as fp:
    exec(fp.read(), version)

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='fpm-dmri',
    version=version['__version__'],
    description='Fourier representation of the diffusion MRI signal using layer potentials method',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/fachra/FourierPotential/',
    author='Chengran Fang',
    author_email='victor.fachra@gmail.com',
    license='GPLv3',

    packages=['fpm'],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'shapely',
        'trimesh',
        'alphashape',
        'torch',
        'torchaudio',
        'torchvision',
        'psutil',
    ],
    include_package_data=True,

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ],
    keywords='diffusion mri, dmri, simulation, potential theory, pytorch',
    project_urls={
    'Source': 'https://github.com/fachra/FourierPotential',
    'Tracker': 'https://github.com/fachra/FourierPotential/issues',
    },
)
