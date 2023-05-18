import os
from setuptools import setup, find_packages

setup(name='open-creativity-scoring',
      packages=["open_scoring"],
      version='2.0.0',
      description="Library for scoring Alternate Uses Task.",
      url="https://github.com/massivetexts/open-scoring",
      author="Peter Organisciak",
      author_email="peter.organisciak@du.edu",
      packages=find_packages(),
      package_data={'': ['assets/*']},
      include_package_data=True,
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
      install_requires=["numpy", "pandas", "sklearn", "spacy", "gensim", 'inflect', 'openai', 'tqdm', 'duckdb', 'pyarrow', 'langchain', 'openai']
)