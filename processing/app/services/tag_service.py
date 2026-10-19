import json

from app.config.paths import CONFIG_DIR
from app.controllers.tag_controller import TagController
from app.models.tag import Tag
from app.services.logging_service import LoggingService

logger = LoggingService.setup_module_logger(__name__, log_filename="tag_service.log")


class TagService:
    """Service for managing standardized tags and tag instances"""

    def __init__(self, config_path: str | None = None):
        """
        Initialize the TagService with configuration path

        Args:
            config_path: Path to the tags.json configuration file
        """
        if config_path is None:
            config_path = CONFIG_DIR.joinpath("tags.json")

        self.config_path = config_path
        self._standardized_tags: dict[str, Tag] = {}
        self._load_standardized_tags_from_json()

    def _load_standardized_tags_from_json(self):
        """
        Load standardized tag definitions from JSON file (read-only catalog).
        These are NOT saved to database - they're just the available templates.
        """
        try:
            with open(self.config_path, "r", encoding="utf-8") as file:
                tags_config = json.load(file)

            if not tags_config:
                logger.warning("Tags configuration file is empty")
                return

            for tag_id, tag_data in tags_config.items():
                # Create Tag instance with the data from JSON (not saved to DB)
                tag_instance = Tag(
                    name=tag_data["name"],
                    description=tag_data["description"],
                    paper_reference=tag_data.get("paper_reference", ""),
                    example=tag_data.get("example", ""),
                    category=tag_data.get("category"),
                    comment="",  # Default empty comment for standardized tags
                    filters=["SK", "RK"],  # Default filters
                )
                self._standardized_tags[tag_id] = tag_instance

            logger.info(
                f"Loaded {len(self._standardized_tags)} standardized tag definitions from JSON"
            )

        except FileNotFoundError:
            logger.warning(f"Tags configuration file not found: {self.config_path}")
            logger.info("No standardized tags available")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in tags configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading standardized tags from config: {e}")
            raise

    def get_standardized_tag_by_name(self, name: str) -> Tag | None:
        """
        Get a standardized tag definition by name (from JSON, not database)

        Args:
            name: The name of the standardized tag

        Returns:
            Tag object or None if not found
        """
        for tag in self._standardized_tags.values():
            if tag.name == name:
                return tag
        return None

    def list_standardized_tags(self) -> dict[str, Tag]:
        """
        Get all available standardized tag definitions (from JSON, not database)

        Returns:
            Dictionary of tag_name -> Tag
        """
        return {tag.name: tag for tag in self._standardized_tags.values()}

    def get_tags_by_category(self, category: str) -> dict[str, Tag]:
        """
        Get standardized tag definitions filtered by category (from JSON, not database)

        Args:
            category: The category to filter by

        Returns:
            Dictionary of tag_name -> Tag for the specified category
        """
        return {
            tag.name: tag
            for tag in self._standardized_tags.values()
            if tag.category == category
        }

    def get_categories(self) -> list[str]:
        """
        Get all unique categories from standardized tag definitions

        Returns:
            List of unique category names
        """
        categories = set()
        for tag in self._standardized_tags.values():
            if tag.category:
                categories.add(tag.category)
        return sorted(categories)

    def validate_tag_name(self, name: str) -> bool:
        """
        Validate if a tag name exists in the standardized tag definitions

        Args:
            name: The tag name to validate

        Returns:
            True if tag name exists, False otherwise
        """
        return self.get_standardized_tag_by_name(name) is not None

    def create_tag_instance(
        self,
        tag_name: str,
        comment: str = "",
        filters: list[str] | None = None,
        job_id: str = "",
    ) -> Tag:
        """
        Create a new tag instance based on a standardized tag and save it to database

        Args:
            tag_name: The standardized tag name to use
            comment: User comment for this tag instance
            filters: List of filters (defaults to ["SK", "RK"])
            job_id: ID of the job this tag is applied to

        Returns:
            Tag instance (saved to database)

        Raises:
            ValueError: If tag_name is not valid
        """
        if filters is None:
            filters = ["SK", "RK"]

        # Get the standardized tag definition by name
        standardized_tag = self.get_standardized_tag_by_name(tag_name)
        if not standardized_tag:
            raise ValueError(f"Invalid tag_name: {tag_name}")

        # Create a new instance with user data and save to database
        # BaseMongoModel will auto-generate the ID
        tag_instance = Tag(
            name=standardized_tag.name,
            description=standardized_tag.description,
            paper_reference=standardized_tag.paper_reference,
            example=standardized_tag.example,
            category=standardized_tag.category,
            comment=comment,
            filters=filters,
            job_id=job_id,
        )

        # Save to database as a tag instance (allows duplicate tag names for different jobs)
        tag_dict = TagController.create_tag_instance(tag_instance)
        return Tag(**tag_dict)

    def search_tags(self, query: str) -> dict[str, Tag]:
        """
        Search standardized tag definitions by name, description, or category

        Args:
            query: Search query string

        Returns:
            Dictionary of matching tag_name -> Tag
        """
        query_lower = query.lower()
        matching_tags = {}

        for tag in self._standardized_tags.values():
            if (
                query_lower in tag.name.lower()
                or query_lower in tag.description.lower()
                or (tag.category and query_lower in tag.category.lower())
            ):
                matching_tags[tag.name] = tag

        return matching_tags

    def reload_configuration(self):
        """
        Reload the standardized tag definitions from the configuration file
        """
        try:
            logger.info("Reloading tag configuration from file")
            self._standardized_tags.clear()
            self._load_standardized_tags_from_json()
            logger.info("Tag configuration reloaded successfully")
        except Exception as e:
            logger.error(f"Error reloading tag configuration: {e}")
            raise

    # Database operations for actual tag instances applied to jobs

    def get_tag_instance_by_id(self, tag_id: str) -> Tag | None:
        """
        Get a tag instance from database by ID

        Args:
            tag_id: The ID of the tag instance

        Returns:
            Tag object or None if not found
        """
        try:
            tag_dict = TagController.get_tag_by_id(tag_id)
            return Tag(**tag_dict)
        except ValueError:
            return None
        except Exception as e:
            logger.error(f"Error getting tag instance {tag_id}: {e}")
            raise

    def get_all_tag_instances(self) -> list[Tag]:
        """
        Get all tag instances from database

        Returns:
            List of Tag instances
        """
        try:
            tag_dicts = TagController.get_all_tags()
            return [Tag(**tag_dict) for tag_dict in tag_dicts]
        except Exception as e:
            logger.error(f"Error getting all tag instances: {e}")
            raise

    def get_tag_instances_by_job(self, job_id: str) -> list[Tag]:
        """
        Get all tag instances for a specific job

        Args:
            job_id: The job ID to filter by

        Returns:
            List of Tag instances for the job
        """
        try:
            # This would need a new method in TagController
            all_tags = TagController.get_all_tags()
            job_tags = [
                Tag(**tag_dict)
                for tag_dict in all_tags
                if tag_dict.get("job_id") == job_id
            ]
            return job_tags
        except Exception as e:
            logger.error(f"Error getting tag instances for job {job_id}: {e}")
            raise

    def update_tag_instance(self, tag_id: str, updates: dict[str, any]) -> Tag:
        """
        Update an existing tag instance in database

        Args:
            tag_id: The ID of the tag instance to update
            updates: Dictionary of fields to update

        Returns:
            Updated Tag instance
        """
        try:
            tag_dict = TagController.update_tag(tag_id, updates)
            return Tag(**tag_dict)
        except Exception as e:
            logger.error(f"Error updating tag instance {tag_id}: {e}")
            raise

    def delete_tag_instance(self, tag_id: str) -> dict[str, str]:
        """
        Delete a tag instance from database

        Args:
            tag_id: The ID of the tag instance to delete

        Returns:
            Success message
        """
        try:
            return TagController.delete_tag(tag_id)
        except Exception as e:
            logger.error(f"Error deleting tag instance {tag_id}: {e}")
            raise


# Singleton instance
_tag_service_instance: TagService | None = None


def get_tag_service() -> TagService:
    """Get the singleton TagService instance"""
    global _tag_service_instance
    if _tag_service_instance is None:
        _tag_service_instance = TagService()
    return _tag_service_instance
