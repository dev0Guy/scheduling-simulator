.PHONY: clean build rebuild

# Remove every generated Cython/build artifact so nothing stale survives.
clean:
	find . -name "src/*.c" -not -path "./build/*" -delete
	find . -name "src/*.html" -delete
	find . -name "src/*.so" -delete
	rm -rf build src/*.egg-info

# Compile the extensions in-place from current .pyx sources.
build:
	python setup.py build_ext --inplace

# Clean, then build from scratch -- use this whenever things feel stale.
rebuild: clean build