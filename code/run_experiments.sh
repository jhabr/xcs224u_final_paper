#!/bin/bash

if [[ -z "${PYTHONPATH}" ]]; then
  export PYTHONPATH=$(pwd)+'code'
fi

python3 experiment/runner.py