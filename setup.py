from setuptools import setup, find_packages

VERSION = '0.0.3' 
DESCRIPTION = 'Pytorch Kernel Transformer'
LONG_DESCRIPTION = 'Kernel attention implementation of Pytorch TransformerEncoderLayer'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="pykernsformer", 
        version=VERSION,
        author="Gokhan Egri",
        author_email="<gegri@g.harvard.edu>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=["torch",
                          "numpy"],
        
        keywords=['pytorch', 'transformer', 'attention'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)