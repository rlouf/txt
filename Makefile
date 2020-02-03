lint:
	black --check txt tests
	isort --check txt tests
	flake8 txt tests --count --ignore=E501,E203,E731,W503 --show-source --statistics
	mypy txt tests

test:
	pytest tests
