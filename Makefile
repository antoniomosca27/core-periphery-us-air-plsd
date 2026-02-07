.PHONY: profile

profile:
	py-spy record --output profile.svg -- python -m tests.performance
