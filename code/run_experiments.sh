#!/bin/bash

if [[ -z "${PYTHONPATH}" ]]; then
  echo "Settings PYTHONPATH to $(pwd)..."
  export PYTHONPATH=$(pwd)
  echo "Done."
fi
(python3 experiment/runner.py 2>&1) >> experiments.log &
