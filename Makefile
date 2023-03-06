PY ?= .venv/bin/python

.PHONY: setup test bench report clean
setup:
	python3 -m venv .venv
	$(PY) -m pip install -e ".[dev]"
test:
	$(PY) -m pytest -q
bench:
	$(PY) benchmarks/bench.py
report:
	$(PY) -m evalforge report --out reports
clean:
	rm -rf build reports .pytest_cache evalforge/_core*.so *.egg-info
