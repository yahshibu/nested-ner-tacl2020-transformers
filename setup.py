from setuptools import setup, find_packages

setup(
    version='0.0.1',
    description='',
    long_description=open("README.md", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude='data'),  # same as name
    license="MIT",
    install_requires=[
        'torch>=1.10.0',
        'numpy>=1.17.3',
        'adabound>=0.0.5',
        'stanfordnlp>=0.2.0',
        'transformers>=4.12.0'
    ],
    python_requires='>=3.7',
)
