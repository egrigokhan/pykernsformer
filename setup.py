import pathlib
from setuptools import setup, find_packages

VERSION = '0.0.4' 
DESCRIPTION = 'Kernel attention implementation of Pytorch TransformerEncoderLayer'

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="pykernsformer", 
        version=VERSION,
        author="Gokhan Egri",
        author_email="<gegri@g.harvard.edu>",
        url="https://github.com/egrigokhan/pykernsformer",
        long_description=README,
        long_description_content_type="text/markdown",
        description=DESCRIPTION,
        packages=find_packages(),
        install_requires=["torch",
                          "numpy"],
       license="MIT",
        
        keywords=['pytorch', 'transformer', 'attention'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)
