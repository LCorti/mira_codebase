import logging

import requests
from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver

from .models import Method, Project, Test
from .services import processing_sync

logger = logging.getLogger("projects.signals")


@receiver(post_save, sender=Project)
def sync_project_on_save(sender, instance, created, **kwargs):
    """
    Automatically sync project to processing service when saved.
    Triggered on both create and update operations.
    """
    if getattr(instance, "_syncing", False):
        return  # Skip if update was made by the sync service itself to prevent recursion
    if instance.sync_status == "synced":
        return  # Only sync if not already synced
    if instance.sync_status == "pending" and not created:
        return  # Avoid concurrent sync operations, but allow sync for newly created projects

    try:
        # Set status to pending before sync attempt using direct database update
        Project.objects.filter(id=instance.id).update(sync_status="pending")

        # Refresh instance to get updated sync_status
        instance.refresh_from_db()

        # Attempt sync to processing service
        success = processing_sync.sync_project_to_processing_service(
            instance, "UPDATE" if not created else "CREATE"
        )

        if not success:
            logger.error(
                f"Failed to sync project {instance.id} after {'creation' if created else 'update'}"
            )
            return

        logger.debug(
            f"Project {instance.id} ({'created' if created else 'updated'}) and synced successfully"
        )

    except Exception as e:
        logger.error(f"Error in sync_project_on_save for project {instance.id}: {e}")
        Project.objects.filter(id=instance.id).update(sync_status="error")


@receiver(post_delete, sender=Project)
def sync_project_on_delete(sender, instance, **kwargs):
    """
    Automatically remove project from processing service when deleted.
    """
    try:
        if instance.id and processing_sync.test_connection():
            response = requests.delete(
                f"{processing_sync.base_url}/projects/{instance.id}",
                timeout=processing_sync.timeout,
            )

            if response.status_code in [200, 204, 404]:  # 404 means already deleted
                logger.info(f"Project {instance.id} deleted from processing service")
            else:
                logger.error(
                    f"Failed to delete project {instance.id} from processing service: {response.status_code}"
                )
        else:
            logger.warning(
                f"Project {instance.id} deleted locally but no processing service ID or connection"
            )

    except Exception as e:
        logger.error(f"Error in sync_project_on_delete for project {instance.id}: {e}")


@receiver(post_save, sender=Test)
def sync_test_on_save(sender, instance, created, **kwargs):
    """
    Update parent project sync status when test is modified locally.
    """
    if getattr(instance, "_syncing", False):
        return  # Skip if update was made by the sync service itself to prevent recursion
    if instance.sync_status == "synced":
        return  # Only sync if not already synced
    if instance.sync_status == "pending" and not created:
        return  # Avoid concurrent sync operations, but allow sync for newly created projects

    try:
        # Set status to pending before sync attempt using direct database update
        Test.objects.filter(id=instance.id).update(sync_status="pending")

        # Refresh instance to get updated sync_status
        instance.refresh_from_db()

        # Attempt sync to processing service
        success = processing_sync.sync_test_to_processing_service(
            instance, "UPDATE" if not created else "CREATE"
        )

        if not success:
            logger.error(
                f"Failed to sync test {instance.id} after {'creation' if created else 'update'}"
            )
            return

        logger.debug(
            f"Test {instance.id} ({'created' if created else 'updated'}) and synced successfully"
        )
    except Exception as e:
        logger.error(f"Error in sync_test_on_save for test {instance.id}: {e}")
        Test.objects.filter(id=instance.id).update(sync_status="error")


@receiver(post_save, sender=Method)
def sync_method_on_save(sender, instance, created, **kwargs):
    """
    Update parent project sync status when method is modified locally.
    """
    if getattr(instance, "_syncing", False):
        return  # Skip if update was made by the sync service itself to prevent recursion

    try:
        # Set status to pending before sync attempt using direct database update
        Method.objects.filter(id=instance.id).update(sync_status="pending")

        # Refresh instance to get updated sync_status
        instance.refresh_from_db()

        # Attempt sync to processing service
        success = processing_sync.sync_method_to_processing_service(instance)

        if not success:
            logger.error(
                f"Failed to sync method {instance.id} after {'creation' if created else 'update'}"
            )
            return

        logger.debug(
            f"Method {instance.id} ({'created' if created else 'updated'}) and synced successfully"
        )
    except Exception as e:
        logger.error(f"Error in sync_method_on_save for method {instance.id}: {e}")
        Method.objects.filter(id=instance.id).update(sync_status="error")
