SHELL := /bin/bash

.PHONY: setup
setup:
	${MAKE} clean
	./setup.sh
	rm -rf .venv
	[[ "$$(python -V)" == "Python 3.7.15" ]]
	poetry install
	poetry run pip install -U pip
	cd external/chamfer && poetry run pip install -e .
	cd external/neural_renderer && poetry run pip install -e .

.PHONY: clean
clean:
#	rm -rf ".wheel"
	rm -rf ".venv"
	rm -rf "external/chamfer/build"
	rm -rf "external/chamfer/*.egg-info"
	rm -rf "external/chamfer/*.so"
	rm -rf "external/neural_renderer/build"
	rm -rf "external/neural_renderer/*.egg-info"
