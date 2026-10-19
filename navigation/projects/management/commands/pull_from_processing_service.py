import sys

from django.core.management.base import BaseCommand

from projects.services import processing_sync


class Command(BaseCommand):
    help = "Pull all projects, tests, and methods from the processing service and create instances in the navigation service"

    def add_arguments(self, parser):
        parser.add_argument(
            "--force",
            action="store_true",
            help="Force the operation even if connection test fails",
        )

        parser.add_argument(
            "--url",
            type=str,
            default=None,
            help="Custom URL for the processing service (overrides default)",
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS("Starting pull operation from processing service...")
        )

        # Debug: Show the URL being used
        url_to_use = options["url"] if options["url"] else "default from settings"
        self.stdout.write(f"Using URL: {url_to_use}")

        # Test connection first unless force is specified
        if not options["force"]:
            self.stdout.write("Testing connection to processing service...")
            if not processing_sync.test_connection(url=options["url"]):
                self.stdout.write(
                    self.style.ERROR(
                        "Failed to connect to processing service. Use --force to bypass this check."
                    )
                )
                sys.exit(1)
            self.stdout.write(self.style.SUCCESS("Connection test successful."))

        try:
            success = processing_sync.pull_all_from_processing_service(
                url=options["url"]
            )

            if success:
                self.stdout.write(
                    self.style.SUCCESS(
                        "Successfully pulled all data from processing service."
                    )
                )
            else:
                self.stdout.write(
                    self.style.ERROR("Failed to pull data from processing service.")
                )
                sys.exit(1)

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"Error pulling data from processing service: {e}")
            )
            sys.exit(1)
