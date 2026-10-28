.PHONY: check

check:
	uv lock --check --offline
	uv run --locked --no-sync python -m compileall -q hello_world iris
	for script in hello_world/*.sh iris/*.sh; do sh -n "$$script" || exit; done
