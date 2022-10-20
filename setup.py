import os
from setuptools import setup

setup(name='open-creativity-scoring',
      packages=["open_scoring"],
      version='1.3.1',
      description="Library for scoring Alternate Uses Task.",
      url="https://github.com/massivetexts/open-scoring",
      author="Peter Organisciak",
      author_email="peter.organisciak@du.edu",
      license="MIT",
      classifiers=[
        'Intended Audience :: Education',
        "Natural Language :: English",
        'License :: OSI Approved :: MIT License',
        "Operating System :: Unix",
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.1',
        "Topic :: Text Processing :: Indexing",
        "Topic :: Text Processing :: Linguistic"
        ],
      install_requires=["numpy", "pandas", "spacy", "gensim", 'inflect', 'openai', 'tqdm']
)