from setuptools import find_packages, setup

setup(
    name="gboost_module",
    version="0.1.4",
    author="Michael Horrell",
    author_email="mthorrell@github.com",
    description="Torch modules using popular boosting libraries",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mthorrell/gboost_module",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
    ],
)
