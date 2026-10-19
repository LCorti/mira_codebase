"""
Test file to verify timezone handling in BaseMongoModel.

This test verifies that the created_at and updated_at timestamps
are correctly set to Europe/Amsterdam timezone.
"""

import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

# Add the app directory to the Python path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.base_model import BaseMongoModel


class TestModel(BaseMongoModel):
    """Test model that inherits from BaseMongoModel"""

    name: str = "test"


def test_timezone_creation():
    """Test that created_at is set to Amsterdam timezone when model is created"""
    # Create a model instance
    model = TestModel()

    # Get current time in Amsterdam timezone for comparison
    amsterdam_now = datetime.now(ZoneInfo("Europe/Amsterdam"))

    # Check that created_at has the correct timezone
    assert model.created_at.tzinfo == ZoneInfo("Europe/Amsterdam"), (
        f"Expected Amsterdam timezone, got {model.created_at.tzinfo}"
    )

    # Check that the time is very close to current time (within 1 second)
    time_diff = abs((amsterdam_now - model.created_at).total_seconds())
    assert time_diff < 1, (
        f"Created time is too far from current time: {time_diff} seconds"
    )

    print(f"> Model created_at: {model.created_at}")
    print(f"> Current Amsterdam time: {amsterdam_now}")
    print(f"> Time difference: {time_diff:.3f} seconds")


def test_timezone_update():
    """Test that updated_at is set to Amsterdam timezone when model is updated"""
    # Create a model instance
    model = TestModel()
    original_updated_at = model.updated_at

    # Wait a small amount and touch the model
    import time

    time.sleep(0.1)
    model.touch()

    # Check that updated_at has the correct timezone
    assert model.updated_at.tzinfo == ZoneInfo("Europe/Amsterdam"), (
        f"Expected Amsterdam timezone, got {model.updated_at.tzinfo}"
    )

    # Check that updated_at was actually updated
    assert model.updated_at > original_updated_at, (
        "updated_at should be newer after touch()"
    )

    print(f"> Original updated_at: {original_updated_at}")
    print(f"> New updated_at: {model.updated_at}")


def test_timezone_comparison():
    """Test timezone comparison with UTC and local time"""
    model = TestModel()

    # Get current time in different timezones
    utc_now = datetime.now(ZoneInfo("UTC"))
    amsterdam_now = datetime.now(ZoneInfo("Europe/Amsterdam"))

    # Amsterdam is typically UTC+1 or UTC+2 (depending on DST)
    # Calculate expected offset
    amsterdam_offset = amsterdam_now.utcoffset().total_seconds() / 3600
    model_offset = model.created_at.utcoffset().total_seconds() / 3600

    print(f"> UTC time: {utc_now}")
    print(f"> Amsterdam time: {amsterdam_now}")
    print(f"> Model created_at: {model.created_at}")
    print(f"> Amsterdam offset from UTC: {amsterdam_offset} hours")
    print(f"> Model offset from UTC: {model_offset} hours")

    # Verify the model has the same offset as Amsterdam timezone
    assert model_offset == amsterdam_offset, (
        f"Model offset ({model_offset}h) should match Amsterdam offset ({amsterdam_offset}h)"
    )


def test_serialization_preserves_timezone():
    """Test that timezone information is preserved during serialization"""
    model = TestModel()

    # Convert to dict using model_dump
    model_dict = model.model_dump()

    print(f"> Model dict keys: {list(model_dict.keys())}")
    print(f"> Full model dict: {model_dict}")

    # Check that the serialized timestamps contain timezone info
    created_at_obj = model_dict["created_at"]
    updated_at_obj = model_dict["updated_at"]

    print(f"> created_at type: {type(created_at_obj)}")
    print(f"> created_at value: {created_at_obj}")
    print(f"> updated_at type: {type(updated_at_obj)}")
    print(f"> updated_at value: {updated_at_obj}")

    # Test JSON serialization using the model's json method
    json_str = model.model_dump_json()
    print(f"> JSON serialization: {json_str}")

    # Check that timezone offset is included in the JSON string
    # Amsterdam timezone should show +01:00 or +02:00 depending on DST
    assert "+01:00" in json_str or "+02:00" in json_str, (
        f"Timezone offset not found in JSON serialization: {json_str}"
    )


if __name__ == "__main__":
    print("Running timezone tests...")
    print("=" * 50)

    try:
        test_timezone_creation()
        print("> Timezone creation test passed\n")

        test_timezone_update()
        print("> Timezone update test passed\n")

        test_timezone_comparison()
        print("> Timezone comparison test passed\n")

        test_serialization_preserves_timezone()
        print("> Timezone serialization test passed\n")

        print("All timezone tests passed!")

    except Exception as e:
        print(f"> Test failed: {e}")
        raise
