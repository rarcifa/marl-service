run-local:
	poetry run marl-local

lint:
	flake8 --statistics --show-source --benchmark --config .flake8 marl