#!/bin/bash
# Seed both databases (processing + navigation) for local setup
# Usage: ./seed_dbs.sh

set -e

echo "Seeding databases..."
echo ""

# Step 1: Seed processing database
echo "[1/2] Seeding processing database..."

# Check if processing container is running
if ! docker ps --format '{{.Names}}' | grep -q '^processing_service$'; then
    echo "ERROR: processing_service container not running"
    exit 1
fi

docker exec -i processing_service uv run python -m app.database.database_seeder

echo "> Processing database seeded"
echo ""

# Step 2: Seed navigation database
echo "[2/2] Seeding navigation database..."

# Check if navigation container is running
if ! docker ps --format '{{.Names}}' | grep -q '^navigation_service$'; then
    echo "ERROR: navigation_service container not running"
    exit 1
fi

docker exec -i navigation_service python manage.py pull_from_processing_service

echo "> Navigation database seeded"
echo ""
echo "Database seeding complete!"
