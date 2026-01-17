#!/bin/bash
echo "Syncing dependencies with uv..."
uv sync

echo "Running Carbon Oracle System..."
uv run -m src.main
