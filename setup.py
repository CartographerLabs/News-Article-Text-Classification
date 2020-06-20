import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="News-Article-Topic-Text-Classification",
    version="0.0.2",
    author="James Stevenson",
    author_email="hi@jamesstevenson.me",
    description="A pre-trained text classification model for identifying the topic of a news article.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/user1342/News-Article-Text-Classification",
    packages=["news_classification"],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GPL-3.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['matplotlib', 'seaborn', 'feedparser', 'pandas', 'sklearn', 'numpy', 'newspaper3k', 'pathlib'],
)
