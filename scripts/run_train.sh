#!/usr/bin/env bash

set -euo pipefail

python -m src.train --config "${1:-configs/teacher_toy.yaml}"

