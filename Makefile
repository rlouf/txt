lint:
	black txt --check .
	flake8 txt --count --ignore=E501,E203,E731,W503 --show-source --statistics
	mypy txt

test:
	pytest tests
