#!/bin/bash
# Seed processing database
# Usage: ./seed_processing.sh

set -e

echo "Seeding processing database..."

# Check if processing container is running
if ! docker ps --format '{{.Names}}' | grep -q '^processing_service$'; then
    echo "ERROR: processing_service container not running"
    exit 1
fi

docker exec -i processing_service uv run python -m app.database.database_seeder

echo "> Processing database seeded"
