import os
from setuptools import setup

setup(name='Open Creativity Scoring',
      packages=["open_scoring"],
      version='1.2.0',
      description="Library for scoring Alternate Users Task.",
      url="https://github.com/massivetexts/open-scoring",
      author="Peter Organisciak",
      author_email="peter.organisciak@du.edu",
      license="MIT",
      classifiers=[
        'Intended Audience :: Education',
        "Natural Language :: English",
        'License :: OSI Approved :: MIT License',
        "Operating System :: Unix",
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.1',
        "Topic :: Text Processing :: Indexing",
        "Topic :: Text Processing :: Linguistic"
        ],
      install_requires=["numpy", "pandas", "spacy", "gensim", 'inflect', 'tqdm']
)
