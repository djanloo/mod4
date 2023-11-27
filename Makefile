.PHONY: generate profile clean

generate:
	@python3 -m mod4.setup

profile:
	# @make clean
	@python3 -m mod4.setup --profile

notrace:
	# @make clean
	@python3 -m mod4.setup --notrace

hardcore:
	# make clean
	@python3 -m mod4.setup --hardcore

hardcoreprofile:
	# make clean
	@python3 -m mod4.setup --hardcore --profile

clean:
	@echo "Cleaning all.."
	@rm -f mod4/*.c
	@rm -f mod4/*.cpp
	@rm -f mod4/*.so
	@rm -f mod4/*.html
	@rm -R -f mod4/build
	@rm -R -f mod4/__pycache__
	@echo "Cleaned."