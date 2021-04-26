import os

from setuptools import setup

path = os.path.abspath(os.path.dirname(__file__))

try:
    with open(os.path.join(path, "requirements.txt"), encoding="utf-8") as f:
        REQUIRED = f.read().split("\n")
except FileNotFoundError:
    REQUIRED = []

setup(
    name="recommender",
    version="0.0.1",
    author="Youness MANSAR",
    author_email="mansaryounessecp@gmail.com",
    description="recommender",
    license="MIT",
    keywords="recommender",
    url="https://github.com/CVxTz/recommender_transformer",
    packages=["recommender"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
    install_requires=REQUIRED,
)
