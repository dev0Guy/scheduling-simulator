.PHONY: clean build rebuild

# Remove every generated Cython/build artifact so nothing stale survives.
clean:
	find src -name "*.c" -delete
	find src -name "*.html" -delete
	find src -name "*.so" -delete
	rm -rf build src/*/*.egg-info

# Compile the extensions in-place from current .pyx sources.
build:
	python setup.py build_ext --inplace

test: 
	pytest -s -v . 

# Clean, then build from scratch -- use this whenever things feel stale.
rebuild: clean build test