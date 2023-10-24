
from setuptools import setup, find_packages

INSTALL_REQUIRES = [
    # generic
    "numpy",
    "torch",
    "prettytable==3.3.0",
    # devices
    "hidapi",
]

# url=EXTENSION_TOML_DATA["package"]["repository"], # add later
# version=EXTENSION_TOML_DATA["package"]["version"], 
# description=EXTENSION_TOML_DATA["package"]["description"],
# keywords=EXTENSION_TOML_DATA["package"]["keywords"],

setup(
    name="omni-orbit-rover",
    author="Anton Bjørndahl Mortensen",
    maintainer="Anton Bjørndahl Mortensen",
    maintainer_email="abmoRobotics@gmail.com",
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=INSTALL_REQUIRES,
    packages=["custom_orbit"],
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.7"],
    zip_safe=False,
)
