.PHONY: setup etl duckdb all

setup:
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt

etl:
	python etl/fetch_bcch_indicators.py
	python etl/fetch_chile_equity.py

duckdb: etl
	python etl/build_duckdb.py

all: duckdb
