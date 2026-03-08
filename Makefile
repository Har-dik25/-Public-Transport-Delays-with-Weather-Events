.PHONY: help install data features train pipeline test dashboard

help:
	@echo "Public Transport Delay Analysis - Command Menu"
	@echo "-----------------------------------------------"
	@echo "make install    : Install Python dependencies"
	@echo "make data       : Run data download & preprocessing"
	@echo "make features   : Run feature engineering"
	@echo "make train      : Train models & forecast"
	@echo "make pipeline   : Run the ENTIRE pipeline (data -> features -> train -> optimize)"
	@echo "make test       : Run pytest unit tests"
	@echo "make dashboard  : Launch the Streamlit dashboard"

install:
	pip install -r requirements.txt

data:
	python -m src.data_loader
	python -m src.preprocessing

features:
	python -m src.feature_engineering

train:
	python -m src.model
	python src/forecasting.py

optimize:
	python scripts/optimize_production.py

pipeline: data features train optimize
	@echo "Full pipeline executed successfully."

test:
	pytest tests/test_pipeline.py -v

dashboard:
	streamlit run dashboard/app.py
