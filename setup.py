from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="pycomo",
    version="0.1.0",
    description="PyCoMo is a software package for generating and analysing compartmentalized community metabolic models",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/univieCUBE/PyCoMo",
    author="Michael Predl",
    author_email="michael.predl@univie.ac.at",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3"
    ],
    entry_points={
        'console_scripts': [
            'pycomo = pycomo.pycomo_models:main'
        ]
    },
    install_requires=["cobra >= 0.23.0", "pandas >= 1.5.3", "python-libsbml >= 5.20.1", "numpy >= 1.22.4"],
    extra_require={
        "dev": ["pytest >= 7.0", "twine >= 4.0.2"],
    },
    python_requires=">=3.9",
)
