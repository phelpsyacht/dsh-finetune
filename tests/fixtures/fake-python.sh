#!/bin/sh
# Stub interpreter for the test suite: satisfies resolveTrainPython's
# PY_CHECK probe without requiring torch. Tests only exercise tool wiring
# (config, cleaned copy, job registration) and never run real training.
echo "3.12.0 torch=0.0.0 transformers=0.0.0 peft=0.0.0"
