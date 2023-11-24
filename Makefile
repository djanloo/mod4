.PHONY: generate profile clear

generate:
	@python3 -m mod4.setup

profile:
	# @make clear
	@python3 -m mod4.setup --profile

notrace:
	# @make clear
	@python3 -m mod4.setup --notrace

hardcore:
	# make clear
	@python3 -m mod4.setup --hardcore

hardcoreprofile:
	# make clear
	@python3 -m mod4.setup --hardcore --profile

clear:
	@echo "Cleaning all.."
	@rm -f mod4/*.c
	@rm -f mod4/*.cpp
	@rm -f mod4/*.so
	@rm -f mod4/*.html
	@rm -R -f mod4/build
	@rm -R -f mod4/__pycache__
	@echo "Cleaned."