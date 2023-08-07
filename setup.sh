#!/bin/bash

python_version=$(python -c 'import sys; print(sys.version_info[:2])')

poetry config virtualenvs.in-project true
poetry lock
poetry install

# Only install dtaidistance for Python 3.9 or lower
if [[ "$python_version" != "(3, 10)" && "$python_version" != "(3, 11)" ]]; then
  poetry run python -m pip install dtaidistance
fi
