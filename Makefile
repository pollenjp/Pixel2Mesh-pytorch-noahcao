SHELL := /bin/bash

.PHONY: setup
setup:
	${MAKE} clean
	./setup.sh
	rm -rf .venv
	poetry config virtualenvs.in-project true
	poetry install

.PHONY: clean
clean:
	rm -rf ".venv"
