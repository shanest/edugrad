import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="edugrad",
    version="0.0.1",
    author="Shane Steinert-Threlkeld",
    author_email="ssshanest@gmail.com",
    description="Basic computation graph, for pedagogical purposes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shanest/edugrad",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=1.17",
        "networkx>=2.4"
    ]
)