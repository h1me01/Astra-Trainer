#!/bin/bash
set -e

BINARY=./build/debug/astra_trainer
ARGS="${@}"

echo "=== memcheck ==="
compute-sanitizer --tool=memcheck --leak-check=full $BINARY $ARGS

echo "=== racecheck ==="
compute-sanitizer --tool=racecheck $BINARY $ARGS

echo "=== synccheck ==="
compute-sanitizer --tool=synccheck $BINARY $ARGS
