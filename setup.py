from setuptools import setup, find_packages

setup(
    version='0.0.1',
    description='',
    long_description=open("README.md", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude='data'),  # same as name
    license="MIT",
    install_requires=[
        'torch>=1.0.1',
        'numpy>=1.16.2',
        'adabound>=0.0.5',
        'pytorch_pretrained_bert>=0.6.1',
        'stanfordnlp>=0.1.2'
    ],
    python_requires='>=3.7',
)
