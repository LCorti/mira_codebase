#!/bin/bash
# Seed navigation database
# Usage: ./seed_navigation.sh

set -e

echo "Seeding navigation database..."

# Check if navigation container is running
if ! docker ps --format '{{.Names}}' | grep -q '^navigation_service$'; then
    echo "ERROR: navigation_service container not running"
    exit 1
fi

docker exec -i navigation_service python manage.py pull_from_processing_service

echo "> Navigation database seeded"
