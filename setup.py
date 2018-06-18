import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()



def find_dependencies():
    deps = ["numpy","nltk"]
    try:
        import tensorflow
    # Only list tensorflow as requirement if not already installed
    except ImportError:
        deps.append("tensorflow")
    return deps



setuptools.setup(
    name="ftodtf",
    version="0.0.1",
    description="Run FastText on distributed TensorFlow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dbaumgarten/FToDTF",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ),
    entry_points = {
        'console_scripts': ['fasttext=ftodtf.cli:cli_main'],
    },
    install_requires = find_dependencies()
)