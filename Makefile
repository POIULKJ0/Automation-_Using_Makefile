.PHONY: setup download-data preprocess features train predict evaluate all clean

setup:
	pip install -r requirements.txt

download-data:
	python src/download_data.py

preprocess:
	python src/preprocess.py

features:
	python src/features.py

train:
	python src/train.py

predict:
	python src/predict.py

evaluate:
	python src/evaluate.py

all: setup download-data preprocess features train predict evaluate

clean:
	rm -rf data/raw/* data/processed/* models/* results/*
