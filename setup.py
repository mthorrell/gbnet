from setuptools import find_packages, setup

setup(
    name="gbnet",
    version="0.6.1",
    author="Michael Horrell",
    author_email="mthorrell@github.com",
    description="Gradient Boosting libraries integrated with PyTorch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mthorrell/gbnet",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
    ],
)
