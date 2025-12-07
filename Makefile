format:
	python -m black --line-length=88 .
	python -m isort --profile black .

lint:
	python -m flake8 .
	python -m mypy src --ignore-missing-imports

test:
	python -m pytest -q

check:
	python -m black --line-length=88 .
	python -m isort --profile black .
	python -m flake8 .
	python -m mypy src --ignore-missing-imports
	python -m pytest -q
