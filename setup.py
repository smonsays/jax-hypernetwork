from setuptools import find_namespace_packages, setup

setup(
    name="jax-hypernetwork",
    version="0.0.1",
    description="A simple hypernetwork implementation in jax using haiku.",
    author="smonsays",
    url="https://github.com/smonsays/jax-hypernetwork",
    license='MIT',
    install_requires=["dm_haiku"],
    packages=find_namespace_packages(),
)
