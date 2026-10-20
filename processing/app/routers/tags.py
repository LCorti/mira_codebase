from typing import ClassVar

from fastapi import APIRouter, HTTPException, Query

from app.models.base_model import BaseMongoModel
from app.models.tag import Tag
from app.services.logging_service import LoggingService
from app.services.tag_service import get_tag_service

logger = LoggingService.setup_module_logger(__name__)

router = APIRouter(prefix="/tags", tags=["tags"])


# Request/Response models
class CreateTagInstanceRequest(BaseMongoModel):
    tag_name: str
    comment: str = ""
    filters: ClassVar[list[str]] = ["SK", "RK"]
    job_id: str = ""


class UpdateTagInstanceRequest(BaseMongoModel):
    comment: str | None = None
    filters: list[str] | None = None
    job_id: str | None = None


@router.get("/available", response_model=dict[str, Tag])
async def get_available_tags(
    category: str | None = Query(None, description="Filter by category"),
):
    """Get all available tag definitions that users can choose from"""
    try:
        tag_service = get_tag_service()

        if category:
            tags = tag_service.get_tags_by_category(category)
        else:
            tags = tag_service.list_standardized_tags()

        return tags
    except Exception as e:
        logger.error(f"Error fetching available tags: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch available tags")


@router.get("/available/{tag_name}", response_model=Tag)
async def get_available_tag(tag_name: str):
    """Get a specific available tag definition by name"""
    try:
        tag_service = get_tag_service()
        tag = tag_service.get_standardized_tag_by_name(tag_name)

        if not tag:
            raise HTTPException(
                status_code=404, detail=f"Available tag '{tag_name}' not found"
            )

        return tag
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching available tag {tag_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch available tag")


@router.get("/instances", response_model=list[Tag])
async def get_tag_instances(
    job_id: str | None = Query(None, description="Filter by job ID"),
):
    """Get tag instances that have been applied to jobs"""
    try:
        tag_service = get_tag_service()

        if job_id:
            tags = tag_service.get_tag_instances_by_job(job_id)
        else:
            tags = tag_service.get_all_tag_instances()

        return tags
    except Exception as e:
        logger.error(f"Error fetching tag instances: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch tag instances")


@router.get("/instances/{tag_id}", response_model=Tag)
async def get_tag_instance(tag_id: str):
    """Get a specific tag instance by ID"""
    try:
        tag_service = get_tag_service()
        tag = tag_service.get_tag_instance_by_id(tag_id)

        if not tag:
            raise HTTPException(
                status_code=404, detail=f"Tag instance '{tag_id}' not found"
            )

        return tag
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching tag instance {tag_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch tag instance")


@router.post("/instances", response_model=Tag)
async def create_tag_instance(request: CreateTagInstanceRequest):
    """Create a new tag instance based on an available tag definition"""
    try:
        tag_service = get_tag_service()
        tag = tag_service.create_tag_instance(
            tag_name=request.tag_name,
            comment=request.comment,
            filters=request.filters,
            job_id=request.job_id,
        )
        return tag
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating tag instance: {e}")
        raise HTTPException(status_code=500, detail="Failed to create tag instance")


@router.put("/instances/{tag_id}", response_model=Tag)
async def update_tag_instance(tag_id: str, request: UpdateTagInstanceRequest):
    """Update an existing tag instance"""
    try:
        tag_service = get_tag_service()

        # Only include fields that are not None
        updates = {}
        if request.comment is not None:
            updates["comment"] = request.comment
        if request.filters is not None:
            updates["filters"] = request.filters
        if request.job_id is not None:
            updates["job_id"] = request.job_id

        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")

        tag = tag_service.update_tag_instance(tag_id, updates)
        return tag
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating tag instance {tag_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update tag instance")


@router.delete("/instances/{tag_id}")
async def delete_tag_instance(tag_id: str):
    """Delete a tag instance"""
    try:
        tag_service = get_tag_service()
        result = tag_service.delete_tag_instance(tag_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting tag instance {tag_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete tag instance")


@router.get("/categories", response_model=list[str])
async def get_tag_categories():
    """Get all available tag categories"""
    try:
        tag_service = get_tag_service()
        return tag_service.get_categories()
    except Exception as e:
        logger.error(f"Error fetching tag categories: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch tag categories")


@router.get("/search", response_model=dict[str, Tag])
async def search_available_tags(
    q: str = Query(
        ..., description="Search query for tag name, description, or category"
    ),
):
    """Search available tag definitions by name, description, or category"""
    try:
        tag_service = get_tag_service()
        return tag_service.search_tags(q)
    except Exception as e:
        logger.error(f"Error searching tags: {e}")
        raise HTTPException(status_code=500, detail="Failed to search tags")


@router.post("/validate-available")
async def validate_available_tag_name(tag_name: str):
    """Validate if a tag name exists in available tag definitions"""
    try:
        tag_service = get_tag_service()
        is_valid = tag_service.validate_tag_name(tag_name)
        return {"tag_name": tag_name, "valid": is_valid}
    except Exception as e:
        logger.error(f"Error validating tag name: {e}")
        raise HTTPException(status_code=500, detail="Failed to validate tag name")


@router.post("/reload-config")
async def reload_tag_configuration():
    """Reload the tag configuration from file (admin function)"""
    try:
        tag_service = get_tag_service()
        tag_service.reload_configuration()
        return {"message": "Tag configuration reloaded successfully"}
    except Exception as e:
        logger.error(f"Error reloading tag configuration: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to reload tag configuration"
        )
