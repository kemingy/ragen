PY_SOURCE=.

lint:
	@black --check --diff ${PY_SOURCE}
	@ruff check .

format:
	@black ${PY_SOURCE}
	@ruff check --fix .

clean:
	@-rm -rf dist build __pycache__ src/*.egg-info ragen/__version__.py

build:
	@python -m build
