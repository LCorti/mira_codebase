#!/bin/bash
# Cleanup script to free disk space on VM

echo "=== Disk Space Cleanup Script ==="
echo "Current disk usage:"
df -h

echo -e "\n=== Cleaning Docker resources ==="
# Remove unused Docker resources
docker system prune -a -f --volumes

echo -e "\n=== Cleaning UV caches ==="
# Clean UV caches older than 30 days
find /data/dtk_storage/docker/uv_cache_nav -type f -mtime +30 -delete 2>/dev/null || true
find /data/dtk_storage/docker/uv_cache_proc -type f -mtime +30 -delete 2>/dev/null || true

echo -e "\n=== Cleaning old reports (older than 90 days) ==="
# Remove old reports to save space
find /data/dtk_storage/docker/reports -type f -mtime +90 -delete 2>/dev/null || true

echo -e "\n=== Cleaning Hugging Face cache duplicates ==="
# Remove duplicate model files (keeping most recent)
# This is safe as models will be re-downloaded if needed
find /data/dtk_storage/docker/hf_cache -type f -name "*.lock" -delete 2>/dev/null || true
find /data/dtk_storage/docker/hf_cache -type d -name "tmp*" -exec rm -rf {} + 2>/dev/null || true

echo -e "\n=== Final disk usage ==="
df -h

echo -e "\n=== Space saved ==="
echo "Cleanup complete!"
