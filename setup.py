import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="local_llm_finetune",
    version="0.0.1",
    author="Daniel Hopp",
    author_email="daniel.hopp@un.org",
    description="Fine-tune an LLM with Unsloth",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dhopp1/local_llm_finetune",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)