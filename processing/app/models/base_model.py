import datetime
from collections.abc import Callable
from enum import Enum
from typing import Any, ClassVar
from zoneinfo import ZoneInfo

from bson import ObjectId
from pydantic import BaseModel, Field, field_validator, model_validator


class BaseMongoModel(BaseModel):
    id: str = Field(default_factory=lambda: str(ObjectId()))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(datetime.UTC),
        description="Timestamp when the record was created (stored in UTC)",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(datetime.UTC),
        description="Timestamp when the record was last updated (stored in UTC)",
    )

    @field_validator("id")
    @classmethod
    def validate_id(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return v

    class Config:
        arbitrary_types_allowed = True
        json_encoders: ClassVar[dict[type, Callable[[Any], Any]]] = {
            Enum: lambda e: e.value,
            datetime: lambda dt: dt.isoformat(),
        }

    def to_dict(self) -> dict:
        """
        Convert to dictionary.

        Args:
            None
        """
        data = self.model_dump(exclude_unset=True)

        for key, value in data.items():
            if isinstance(value, Enum):
                data[key] = value.value
        return data

    def get_local_created_at(self, timezone_name: str = "Europe/Amsterdam") -> datetime:
        """
        Get created_at in the specified timezone (default: Amsterdam).

        Args:
            timezone_name: Target timezone name (e.g., "Europe/Amsterdam")

        Returns:
            datetime: Timestamp in the specified timezone
        """
        if self.created_at.tzinfo is None:
            # Assume stored time is UTC if timezone-naive (from MongoDB)
            utc_time = self.created_at.replace(tzinfo=datetime.UTC)
        else:
            # Convert to UTC first if it has timezone info
            utc_time = self.created_at.astimezone(datetime.UTC)

        return utc_time.astimezone(ZoneInfo(timezone_name))

    def get_local_updated_at(self, timezone_name: str = "Europe/Amsterdam") -> datetime:
        """
        Get updated_at in the specified timezone (default: Amsterdam).

        Args:
            timezone_name: Target timezone name (e.g., "Europe/Amsterdam")

        Returns:
            datetime: Timestamp in the specified timezone
        """
        if self.updated_at.tzinfo is None:
            # Assume stored time is UTC if timezone-naive (from MongoDB)
            utc_time = self.updated_at.replace(tzinfo=datetime.UTC)
        else:
            # Convert to UTC first if it has timezone info
            utc_time = self.updated_at.astimezone(datetime.UTC)

        return utc_time.astimezone(ZoneInfo(timezone_name))

    @classmethod
    def from_mongo(cls, data: dict):
        """
        Create model from MongoDB data, handling timezone conversion for naive datetimes.

        Args:
            data: Dictionary from MongoDB

        Returns:
            BaseMongoModel: Model instance with proper timezone handling
        """
        # Handle MongoDB's timezone-naive UTC datetimes
        if (
            "created_at" in data
            and isinstance(data["created_at"], datetime)
            and data["created_at"].tzinfo is None
        ):
            data["created_at"] = data["created_at"].replace(tzinfo=datetime.UTC)

        if (
            "updated_at" in data
            and isinstance(data["updated_at"], datetime)
            and data["updated_at"].tzinfo is None
        ):
            data["updated_at"] = data["updated_at"].replace(tzinfo=datetime.UTC)

        return cls(**data)

    @model_validator(mode="before")
    @classmethod
    def set_updated_at(cls, values):
        """Automatically set updated_at when model is being updated (not on creation)"""
        if (
            isinstance(values, dict)
            and "created_at" in values
            and values.get("created_at") is not None
        ):
            # If created_at exists but updated_at is being set, this is likely an update
            values["updated_at"] = datetime.now(datetime.UTC)
        return values

    def touch(self) -> None:
        """Manually update the updated_at timestamp (in UTC)"""
        self.updated_at = datetime.now(datetime.UTC)

    def model_copy_with_update(self, **kwargs):
        """Override model_copy to automatically update the timestamp (in UTC)"""
        kwargs["updated_at"] = datetime.now(datetime.UTC)
        return super().model_copy(update=kwargs)
