#!/usr/bin/env bash
set -euo pipefail

pytest tests/unit tests/integration "$@"
