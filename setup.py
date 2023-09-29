from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='iclik',
    version='0.06-alpha',
    packages=['iclik'],
    long_description=long_description,
    long_description_content_type="text/markdown"
)
