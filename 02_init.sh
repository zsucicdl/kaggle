#!/bin/bash

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install deepspeed
pip install datasets
python -m scripts/setup/get_data.py
python scripts/setup/get_datasets.py
python -m spacy download en
python -m nltk.downloader stopwords
