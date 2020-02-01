import setuptools

setuptools.setup(
    name="txt",
    version="0.0.1",
    author="Remi Louf",
    author_email="remilouf@gmail.com",
    description="Natural Language Generation made simple",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/remilouf/txt",
    licence="GNU General Public Licence v3.0",
    packages=["txt"],
    python_requires=">=3.5",
    install_requires=["numpy", "torch>=1.3.1", "transformers==2.3.0"],
)
