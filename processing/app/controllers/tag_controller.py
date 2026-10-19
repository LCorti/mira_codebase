from typing import Dict, Any, List, Optional
from contextlib import contextmanager

from app.services.logging_service import LoggingService

from app.models.tag import Tag

logger = LoggingService.setup_module_logger(__name__, log_filename="controllers.log")

def _get_tags_collection():
    """
    Returns the tags collection from the database service.
    """
    from app.config.registry import get_database_service
    db_service = get_database_service()
    return db_service.get_collection("tags")

def _get_jobs_collection():
    """
    Returns the jobs collection from the database service.
    """
    from app.config.registry import get_database_service
    db_service = get_database_service()
    return db_service.get_collection("jobs")

class TagController:
    """
    Controller class for handling tag-related business logic.
    """
    
    @staticmethod
    @contextmanager
    def database_transaction():
        """
        Context manager for database transactions to reduce repetition.
        """
        from app.config.registry import get_database_service
        session = get_database_service().start_session()
        try:
            with session.start_transaction():
                yield session
                session.commit_transaction()
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            raise
    
    @staticmethod
    def get_tag_by_id(tag_id: str) -> Dict[str, Any]:
        """
        Retrieves a specific tag by its ID.

        Args:
            tag_id (str): The ID of the tag to retrieve.

        Returns:
            dict: The tag serialized as a dictionary.

        Raises:
            RuntimeError: If there is an error retrieving the tag from the database.
        """
        try:
            tag = _get_tags_collection().find_one({"id": str(tag_id)})
            if not tag:
                logger.warning(f"Tag with ID {tag_id} not found.")
                raise ValueError(f"Tag with ID {tag_id} not found.")
            return Tag(**tag).to_dict()
        except Exception as e:
            logger.error(f"Error retrieving tag with ID {tag_id}: {e}")
            raise RuntimeError(f"Failed to retrieve tag with ID {tag_id}: {e}")
    
    @staticmethod
    def get_tag_by_name(name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a specific tag by its name.

        Args:
            name (str): The name of the tag to retrieve.

        Returns:
            dict: The tag serialized as a dictionary, or None if not found.

        Raises:
            RuntimeError: If there is an error retrieving the tag from the database.
        """
        try:
            tag = _get_tags_collection().find_one({"name": name})
            if not tag:
                return None
            return Tag(**tag).to_dict()
        except Exception as e:
            logger.error(f"Error retrieving tag with name {name}: {e}")
            raise RuntimeError(f"Failed to retrieve tag with name {name}: {e}")
    
    @staticmethod
    def get_all_tags() -> List[Dict[str, Any]]:
        """
        Retrieves all tags from the database.

        Returns:
            list: List of all tags serialized as dictionaries.

        Raises:
            RuntimeError: If there is an error retrieving tags from the database.
        """
        try:
            tags_cursor = _get_tags_collection().find()
            tags = []
            for tag_data in tags_cursor:
                tag = Tag(**tag_data)
                tags.append(tag.to_dict())
            return tags
        except Exception as e:
            logger.error(f"Error retrieving all tags: {e}")
            raise RuntimeError(f"Failed to retrieve all tags: {e}")
    
    @staticmethod
    def get_tags_by_category(category: str) -> List[Dict[str, Any]]:
        """
        Retrieves tags filtered by category.

        Args:
            category (str): The category to filter by.

        Returns:
            list: List of tags in the specified category.

        Raises:
            RuntimeError: If there is an error retrieving tags from the database.
        """
        try:
            tags_cursor = _get_tags_collection().find({"category": category})
            tags = []
            for tag_data in tags_cursor:
                tag = Tag(**tag_data)
                tags.append(tag.to_dict())
            return tags
        except Exception as e:
            logger.error(f"Error retrieving tags by category {category}: {e}")
            raise RuntimeError(f"Failed to retrieve tags by category {category}: {e}")
    
    @staticmethod
    def search_tags(query: str) -> List[Dict[str, Any]]:
        """
        Search tags by name, description, or category using text search.

        Args:
            query (str): The search query.

        Returns:
            list: List of matching tags.

        Raises:
            RuntimeError: If there is an error searching tags in the database.
        """
        try:
            # Use MongoDB text search if available, otherwise use regex
            search_filter = {
                "$or": [
                    {"name": {"$regex": query, "$options": "i"}},
                    {"description": {"$regex": query, "$options": "i"}},
                    {"category": {"$regex": query, "$options": "i"}}
                ]
            }
            
            tags_cursor = _get_tags_collection().find(search_filter)
            tags = []
            for tag_data in tags_cursor:
                tag = Tag(**tag_data)
                tags.append(tag.to_dict())
            return tags
        except Exception as e:
            logger.error(f"Error searching tags with query '{query}': {e}")
            raise RuntimeError(f"Failed to search tags with query '{query}': {e}")
    
    @staticmethod
    def get_categories() -> List[str]:
        """
        Get all unique categories from tags.

        Returns:
            list: List of unique category names.

        Raises:
            RuntimeError: If there is an error retrieving categories from the database.
        """
        try:
            categories = _get_tags_collection().distinct("category")
            # Filter out None values and sort
            categories = [cat for cat in categories if cat is not None]
            return sorted(categories)
        except Exception as e:
            logger.error(f"Error retrieving tag categories: {e}")
            raise RuntimeError(f"Failed to retrieve tag categories: {e}")
    
    @staticmethod
    def create_tag_instance(tag: Tag) -> Dict[str, Any]:
        """
        Creates a new tag instance in the database (allows duplicates by name since instances can be applied multiple times).

        Args:
            tag (Tag): The tag instance data to create.

        Returns:
            dict: The created tag instance serialized as a dictionary.

        Raises:
            RuntimeError: If there is an error creating the tag instance in the database.
        """
        try:
            with TagController.database_transaction():
                tag_insert = _get_tags_collection().insert_one(tag.to_dict())
                if not tag_insert.acknowledged:
                    logger.error(f"Failed to create tag instance.")
                    raise RuntimeError(f"Failed to create tag instance.")
                
                # Update corresponding job to include this tag instance
                from app.models.job import Job
                job_collection = _get_jobs_collection()
                job = job_collection.find_one({"id": str(tag.job_id)})
                if not job:
                    logger.error(f"Job with ID {tag.job_id} not found for tagging.")
                    raise ValueError(f"Job with ID {tag.job_id} not found.")
                
                job_obj = Job(**job)
                if tag.id in job_obj.tag_ids:
                    logger.info(f"Tag ID {tag.id} already associated with Job ID {tag.job_id}.")
                    raise ValueError(f"Tag ID {tag.id} already associated with Job ID {tag.job_id}.")
                
                job_obj.tag_ids.append(tag.id)
                job_collection.update_one(
                    {"id": str(tag.job_id)},
                    {"$set": {"tag_ids": job_obj.tag_ids}}
                )
                    
                return tag.to_dict()
        except Exception as e:
            logger.error(f"Error creating tag instance: {e}")
            raise RuntimeError(f"Failed to create tag instance: {e}")
    
    @staticmethod
    def create_tag(tag: Tag) -> Dict[str, Any]:
        """
        Creates a new tag in the database.

        Args:
            tag (Tag): The tag data to create.

        Returns:
            dict: The created tag serialized as a dictionary.

        Raises:
            RuntimeError: If there is an error creating the tag in the database.
        """
        try:
            with TagController.database_transaction():
                # Check if tag with same name already exists
                existing_tag = _get_tags_collection().find_one({"name": tag.name})
                if existing_tag:
                    raise ValueError(f"Tag with name '{tag.name}' already exists.")
                
                tag_insert = _get_tags_collection().insert_one(tag.to_dict())
                if not tag_insert.acknowledged:
                    logger.error(f"Failed to create tag.")
                    raise RuntimeError(f"Failed to create tag.")
                return tag.to_dict()
        except Exception as e:
            logger.error(f"Error creating tag: {e}")
            raise RuntimeError(f"Failed to create tag: {e}")
    
    @staticmethod
    def update_tag(tag_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Updates an existing tag in the database.

        Args:
            tag_id (str): The ID of the tag to update.
            updates (dict): The fields to update.

        Returns:
            dict: The updated tag serialized as a dictionary.

        Raises:
            RuntimeError: If there is an error updating the tag in the database.
        """
        try:
            with TagController.database_transaction():
                # Add updated_at timestamp
                from datetime import datetime, timezone
                updates['updated_at'] = datetime.now(timezone.utc)
                
                update_result = _get_tags_collection().update_one(
                    {"id": str(tag_id)},
                    {"$set": updates}
                )
                
                if update_result.matched_count == 0:
                    logger.warning(f"Tag with ID {tag_id} not found for update.")
                    raise ValueError(f"Tag with ID {tag_id} not found.")
                
                # Return the updated tag
                updated_tag = _get_tags_collection().find_one({"id": str(tag_id)})
                return Tag(**updated_tag).to_dict()
        except Exception as e:
            logger.error(f"Error updating tag with ID {tag_id}: {e}")
            raise RuntimeError(f"Failed to update tag with ID {tag_id}: {e}")

    @staticmethod
    def delete_tag(tag_id: str):
        """
        Deletes a tag from the database.

        Args:
            tag_id (str): The ID of the tag to delete.

        Returns:
            dict: A message indicating the deletion status.

        Raises:
            RuntimeError: If there is an error deleting the tag from the database.
        """
        try:
            with TagController.database_transaction():
                delete_result = _get_tags_collection().delete_one({"id": str(tag_id)})
                if delete_result.deleted_count == 0:
                    logger.warning(f"Tag with ID {tag_id} not found for deletion.")
                    raise ValueError(f"Tag with ID {tag_id} not found.")

                return {"message": f"Tag with ID {tag_id} deleted successfully."}
        except Exception as e:
            logger.error(f"Error deleting tag with ID {tag_id}: {e}")
            raise RuntimeError(f"Failed to delete tag with ID {tag_id}: {e}")
    
    @staticmethod
    def bulk_create_tags(tags: List[Tag]) -> List[Dict[str, Any]]:
        """
        Creates multiple tags in the database.

        Args:
            tags (List[Tag]): List of tag objects to create.

        Returns:
            list: List of created tags serialized as dictionaries.

        Raises:
            RuntimeError: If there is an error creating tags in the database.
        """
        try:
            with TagController.database_transaction():
                # Check for existing tags with same names
                tag_names = [tag.name for tag in tags]
                existing_tags = list(_get_tags_collection().find({"name": {"$in": tag_names}}))
                existing_names = [tag["name"] for tag in existing_tags]
                
                if existing_names:
                    raise ValueError(f"Tags with names {existing_names} already exist.")
                
                tag_dicts = [tag.to_dict() for tag in tags]
                insert_result = _get_tags_collection().insert_many(tag_dicts)
                
                if not insert_result.acknowledged:
                    logger.error(f"Failed to create tags in bulk.")
                    raise RuntimeError(f"Failed to create tags in bulk.")
                
                return tag_dicts
        except Exception as e:
            logger.error(f"Error creating tags in bulk: {e}")
            raise RuntimeError(f"Failed to create tags in bulk: {e}")