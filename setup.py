from setuptools import setup, find_packages

setup(
    name='authorstyle',
    version='0.2',
    description='Author Style Framework',
    url='https://github.com/mullerpeter/authorstyle',
    author='Peter Muller',
    author_email='pemuelle@student.ethz.ch',
    license='',
    packages=find_packages(exclude=['tests']),
    setup_requires=['setuptools>=38.6.0'],
    include_package_data=True,
    package_data={
        'authorstyle': ['features/external_data/*.txt'],
    },
    install_requires=['scikit-learn', 'textstat', 'numpy', 'nltk', 'pandas', 'cophi', 'tqdm']
)
