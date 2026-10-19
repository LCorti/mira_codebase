"""
MongoDB Timezone Diagnostic Script

This script comprehensively tests timezone handling in MongoDB storage
to diagnose why created_at timestamps might be appearing 2 hours behind.
"""

import datetime
import os
import sys
from zoneinfo import ZoneInfo

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.base_model import BaseMongoModel


class DiagnosticModel(BaseMongoModel):
    """Test model for diagnostic purposes"""

    name: str = "diagnostic_test"
    test_field: str = "testing"


class MongoTimezoneDiagnostic:
    def __init__(
        self,
        connection_string="mongodb://localhost:27017",
        database_name="diagnosis_test",
    ):
        self.connection_string = connection_string
        self.database_name = database_name
        self.client = None
        self.db = None
        self.collection_name = "timezone_test"

    def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = MongoClient(
                self.connection_string, serverSelectionTimeoutMS=5000
            )
            # Test connection
            self.client.server_info()
            self.db = self.client[self.database_name]
            print(f"> Connected to MongoDB at {self.connection_string}")
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"> Failed to connect to MongoDB: {e}")
            return False

    def cleanup(self):
        """Clean up test data"""
        if self.db is not None:
            self.db[self.collection_name].drop()
            print(f"> Cleaned up test collection: {self.collection_name}")

    def test_system_timezone_info(self):
        """Test system timezone information"""
        print("\n" + "=" * 60)
        print("SYSTEM TIMEZONE DIAGNOSTIC")
        print("=" * 60)

        # Get current time in various timezones
        utc_now = datetime.now(datetime.UTC)
        amsterdam_now = datetime.now(ZoneInfo("Europe/Amsterdam"))
        local_now = datetime.now()
        local_tz_now = datetime.now().astimezone()

        print(f"UTC time:                {utc_now}")
        print(f"Amsterdam time:          {amsterdam_now}")
        print(f"Local time (naive):      {local_now}")
        print(f"Local time (with tz):    {local_tz_now}")

        # Calculate offsets
        amsterdam_offset = amsterdam_now.utcoffset().total_seconds() / 3600
        local_offset = local_tz_now.utcoffset().total_seconds() / 3600

        print(f"Amsterdam offset:        {amsterdam_offset:+.1f} hours from UTC")
        print(f"Local timezone offset:   {local_offset:+.1f} hours from UTC")
        print(f"Local timezone name:     {local_tz_now.tzinfo}")

        return {
            "utc_time": utc_now,
            "amsterdam_time": amsterdam_now,
            "local_time": local_now,
            "local_tz_time": local_tz_now,
            "amsterdam_offset": amsterdam_offset,
            "local_offset": local_offset,
        }

    def test_model_creation(self):
        """Test BaseMongoModel timezone behavior"""
        print("\n" + "=" * 60)
        print("MODEL CREATION DIAGNOSTIC")
        print("=" * 60)

        # Create multiple models to see consistency
        models = []
        for i in range(3):
            model = DiagnosticModel(name=f"test_{i}")
            models.append(model)
            print(f"Model {i + 1}:")
            print(f"  created_at:  {model.created_at}")
            print(f"  updated_at:  {model.updated_at}")
            print(f"  timezone:    {model.created_at.tzinfo}")
            print(f"  UTC offset:  {model.created_at.utcoffset()}")

        return models

    def test_direct_mongodb_storage(self, model):
        """Test direct MongoDB storage behavior"""
        print("\n" + "=" * 60)
        print("DIRECT MONGODB STORAGE DIAGNOSTIC")
        print("=" * 60)

        collection = self.db[self.collection_name]

        # Test 1: Store the model data directly
        model_dict = model.model_dump()
        print(f"Model dict before storage: {model_dict}")

        # Insert into MongoDB
        result = collection.insert_one(model_dict)
        inserted_id = result.inserted_id
        print(f"> Inserted document with ID: {inserted_id}")

        # Retrieve from MongoDB
        retrieved_doc = collection.find_one({"_id": inserted_id})
        print(f"Retrieved document: {retrieved_doc}")

        # Compare timestamps
        original_created_at = model.created_at
        stored_created_at = retrieved_doc["created_at"]

        print("\nTimestamp comparison:")
        print(
            f"Original created_at:  {original_created_at} (type: {type(original_created_at)})"
        )
        print(
            f"Stored created_at:    {stored_created_at} (type: {type(stored_created_at)})"
        )

        if isinstance(stored_created_at, datetime):
            if stored_created_at.tzinfo:
                print(f"Stored timezone:      {stored_created_at.tzinfo}")
                print(f"Stored UTC offset:    {stored_created_at.utcoffset()}")

                # Calculate time difference
                time_diff = (original_created_at - stored_created_at).total_seconds()
                print(f"Time difference:      {time_diff} seconds")
            else:
                print("> Stored datetime is timezone-naive!")
                # Assume stored time is in UTC and compare
                stored_utc = stored_created_at.replace(tzinfo=datetime.UTC)
                time_diff = (original_created_at - stored_utc).total_seconds()
                print(f"Time diff (assuming UTC): {time_diff} seconds")

        return retrieved_doc

    def test_mongodb_timezone_settings(self):
        """Test MongoDB server timezone settings"""
        print("\n" + "=" * 60)
        print("MONGODB SERVER TIMEZONE DIAGNOSTIC")
        print("=" * 60)

        try:
            # Get server status
            server_status = self.db.command("serverStatus")
            print(f"MongoDB version: {server_status.get('version', 'Unknown')}")

            # Test current time on server
            server_time_result = self.db.command("eval", "new Date()")
            print(f"MongoDB server time: {server_time_result}")

            # Test storing different datetime formats
            collection = self.db[self.collection_name]

            test_docs = [
                {"test_type": "utc_datetime", "timestamp": datetime.now(datetime.UTC)},
                {
                    "test_type": "amsterdam_datetime",
                    "timestamp": datetime.now(ZoneInfo("Europe/Amsterdam")),
                },
                {"test_type": "naive_datetime", "timestamp": datetime.now()},
            ]

            for doc in test_docs:
                print(f"\nTesting {doc['test_type']}:")
                print(
                    f"  Original: {doc['timestamp']} (type: {type(doc['timestamp'])})"
                )

                # Insert and retrieve
                result = collection.insert_one(doc)
                retrieved = collection.find_one({"_id": result.inserted_id})

                print(
                    f"  Stored:   {retrieved['timestamp']} (type: {type(retrieved['timestamp'])})"
                )

                if isinstance(retrieved["timestamp"], datetime):
                    if retrieved["timestamp"].tzinfo:
                        print(f"  Timezone: {retrieved['timestamp'].tzinfo}")
                    else:
                        print("  Timezone: None (naive)")

        except Exception as e:
            print(f"> Error testing MongoDB server settings: {e}")

    def test_bson_datetime_handling(self):
        """Test how BSON handles datetime objects"""
        print("\n" + "=" * 60)
        print("BSON DATETIME HANDLING DIAGNOSTIC")
        print("=" * 60)

        from bson import decode, encode
        from bson.codec_options import CodecOptions

        # Test different datetime formats
        test_datetimes = {
            "UTC": datetime.now(datetime.UTC),
            "Amsterdam": datetime.now(ZoneInfo("Europe/Amsterdam")),
            "Naive": datetime.now(),
        }

        for name, dt in test_datetimes.items():
            print(f"\nTesting {name} datetime:")
            print(f"  Original:  {dt}")
            print(f"  Timezone:  {dt.tzinfo}")

            # Encode to BSON
            doc = {"timestamp": dt}
            bson_data = encode(doc)

            # Decode from BSON
            decoded_doc = decode(bson_data)
            decoded_dt = decoded_doc["timestamp"]

            print(f"  Decoded:   {decoded_dt}")
            print(f"  Decoded tz: {decoded_dt.tzinfo}")

            # Test with timezone-aware codec options
            codec_options = CodecOptions(tz_aware=True, tzinfo=datetime.UTC)
            decoded_doc_tz = decode(bson_data, codec_options=codec_options)
            decoded_dt_tz = decoded_doc_tz["timestamp"]

            print(f"  Decoded (tz-aware): {decoded_dt_tz}")
            print(f"  Decoded tz (tz-aware): {decoded_dt_tz.tzinfo}")

    def run_full_diagnostic(self):
        """Run the complete diagnostic suite"""
        print("> MONGODB TIMEZONE DIAGNOSTIC SUITE")
        print("> " + "=" * 58)

        # System timezone tests
        system_info = self.test_system_timezone_info()

        # Model creation tests
        models = self.test_model_creation()

        # Connect to MongoDB
        if self.connect():
            try:
                # MongoDB storage tests
                if models:
                    _ = self.test_direct_mongodb_storage(models[0])

                # MongoDB server tests
                self.test_mongodb_timezone_settings()

                # BSON handling tests
                self.test_bson_datetime_handling()

            finally:
                # Cleanup
                self.cleanup()
                if self.client:
                    self.client.close()

        print("\n" + "DIAGNOSTIC SUMMARY")
        print("=" * 50)

        amsterdam_time = system_info["amsterdam_time"]
        local_time = system_info["local_tz_time"]

        print(f"Current Amsterdam time: {amsterdam_time}")
        print(f"Current local time:     {local_time}")
        print(f"Amsterdam offset:       {system_info['amsterdam_offset']:+.1f} hours")
        print(f"Local offset:           {system_info['local_offset']:+.1f} hours")

        if models:
            model_time = models[0].created_at
            print(f"Model created_at:       {model_time}")

            # Check if there's a 2-hour difference
            time_diff_hours = (amsterdam_time - model_time).total_seconds() / 3600
            print(f"Time difference:        {time_diff_hours:.2f} hours")

            if abs(time_diff_hours) > 0.1:
                print("> WARNING: Significant time difference detected!")
            else:
                print("> Model timestamps appear correct")

        print("- Check MongoDB's timezone configuration")
        print("- Ensure your application consistently uses timezone-aware datetimes")
        print("- Consider storing all timestamps in UTC and converting on display")


if __name__ == "__main__":
    # You can modify these connection parameters
    diagnostic = MongoTimezoneDiagnostic(
        connection_string="mongodb://localhost:27017",  # Adjust as needed
        database_name="diagnosis_test",
    )

    diagnostic.run_full_diagnostic()
