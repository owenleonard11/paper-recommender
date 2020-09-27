import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="papers",
    version="0.0.1",
    author="owenleonard11",
    author_email="owenleonard11@gmail.com",
    description="A package used for retrieving and manipulating data from call-for-papers.sas.upenn.edu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/owenleonard11/recommend-papers",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Academics",
    ],
    install_requires=[
        "beautifulsoup4",
        "requests"
    ],
    python_requires=">=3"
)
