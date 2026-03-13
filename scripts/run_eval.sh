#!/usr/bin/env bash

set -euo pipefail

python -m src.eval --config "${1:-configs/teacher_toy.yaml}" "${@:2}"

