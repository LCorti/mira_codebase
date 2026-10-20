import logging
import os
import re
from typing import ClassVar

from app.config.paths import LOGS_DIR


class LoggingService:
    """
    Centralized logging service that creates module-specific loggers.
    Each logger writes to its own log file and optionally to console.
    """

    _loggers: ClassVar[dict[str, logging.Logger]] = {}  # Cache for created loggers

    def __init__(self) -> None:
        if not LOGS_DIR.exists():
            LOGS_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_logger(
        cls,
        name: str,
        log_filename: str | None = None,
        console_level: str = "INFO",
        file_level: str = "DEBUG",
    ) -> logging.Logger:
        """
        Get or create a module-specific logger.

        Args:
            name: Logger name (typically __name__ or module name)
            log_filename: Custom log filename (defaults to module name + .log)
            console_level: Logging level for console output
            file_level: Logging level for file output

        Returns:
            Configured logger instance
        """
        # Use cached logger if it exists
        if name in cls._loggers:
            return cls._loggers[name]

        # Create new logger
        logger = logging.getLogger(name)

        # Prevent adding handlers multiple times
        if logger.handlers:
            cls._loggers[name] = logger
            return logger

        # Ensure logs directory exists
        if not LOGS_DIR.exists():
            LOGS_DIR.mkdir(parents=True, exist_ok=True)

        # Get logging level from environment
        env_level = os.getenv("LOGGING_LEVEL", "INFO")
        logger.setLevel(env_level)

        # Create log filename if not provided
        if log_filename is None:
            # Extract module name from full name (e.g., 'app.services.model_loader' -> 'model_loader')
            module_name = name.split(".")[-1] if "." in name else name
            log_filename = f"{module_name}.log"

        # Ensure log filename has .log extension
        if not log_filename.endswith(".log"):
            log_filename += ".log"

        log_file = LOGS_DIR.joinpath(log_filename)

        # Create formatters
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # File handler
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Prevent propagation to root logger to avoid duplicate messages
        logger.propagate = False

        # Cache the logger
        cls._loggers[name] = logger

        return logger

    @classmethod
    def setup_module_logger(cls, module_name: str, **kwargs) -> logging.Logger:
        """
        Convenience method to set up a logger for a module using __name__.

        :param module_name: Name of the module (typically __name__)
        :param kwargs: Additional keyword arguments for logger configuration

        :Returns:
            Configured logger instance
        :Example:
            logger = LoggingService.setup_module_logger(__name__)
        """
        return cls.get_logger(module_name, **kwargs)

    @classmethod
    def create_custom_logger(
        cls, name: str, log_file: str, level: str = "INFO"
    ) -> logging.Logger:
        """
        Create a custom logger with specific configuration.

        Args:
            name: Logger name
            log_file: Log file name
            level: Logging level

        Returns:
            Configured logger instance
        """
        return cls.get_logger(
            name=name, log_filename=log_file, console_level=level, file_level="DEBUG"
        )

    @classmethod
    def clear_cache(cls):
        """Clear the logger cache. Useful for testing or reinitialization."""
        cls._loggers.clear()

    @classmethod
    def get_job_logs(
        cls, job_id: str, log_filename: str = "job.log", include_timestamp: bool = True
    ) -> list[dict[str, str]]:
        """
        Retrieve all log lines that contain a specific job ID.

        Args:
            job_id: The job ID to search for in log lines
            log_filename: The log file to search in (defaults to job.log)
            include_timestamp: Whether to include timestamp in the parsed result

        Returns:
            List of dictionaries containing parsed log information:
            - timestamp: Log timestamp (if include_timestamp=True)
            - level: Log level (INFO, ERROR, WARNING, etc.)
            - message: Log message content
            - raw_line: Original log line

        Example:
            job_logs = LoggingService.get_job_logs("68adb3e4f708fe4ffde3e957")
            for log in job_logs:
                print(f"[{log['level']}] {log['message']}")
        """
        log_file_path = LOGS_DIR / log_filename
        job_logs = []

        # Pattern to match log lines with job ID
        # Looking for: timestamp - module - level - job_id: message OR timestamp - module - level - message containing job_id
        log_pattern = re.compile(
            r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - "  # timestamp
            r"([\w\.]+) - "  # module name
            r"(\w+) - "  # log level
            r"(.+)$"  # message (may or may not start with job_id:)
        )

        try:
            with open(log_file_path, "r", encoding="utf-8") as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:
                        continue

                    # Check if the line contains the job ID
                    if job_id in line:
                        match = log_pattern.match(line)
                        if match:
                            timestamp, module, level, full_message = match.groups()

                            # Check if message starts with job_id: (like INFO logs)
                            if full_message.startswith(f"{job_id}:"):
                                message = full_message[len(job_id) + 1 :].strip()
                            # Check if message contains job_id anywhere (like WARNING logs)
                            elif job_id in full_message:
                                message = full_message
                            else:
                                continue  # Skip if job_id not found in expected places

                            log_entry = {
                                "level": level,
                                "message": message,
                                "raw_line": line,
                                "line_number": line_num,
                            }

                            if include_timestamp:
                                log_entry["timestamp"] = timestamp
                                log_entry["module"] = module

                            job_logs.append(log_entry)
                        else:
                            # Handle lines that contain job_id but don't match the pattern
                            # (e.g., multi-line stack traces)
                            if job_id in line:
                                log_entry = {
                                    "level": "UNKNOWN",
                                    "message": line,
                                    "raw_line": line,
                                    "line_number": line_num,
                                }

                                if include_timestamp:
                                    log_entry["timestamp"] = "UNKNOWN"
                                    log_entry["module"] = "UNKNOWN"

                                job_logs.append(log_entry)

        except FileNotFoundError:
            # Return empty list if file doesn't exist
            return []
        except PermissionError:
            # Return error information if file cannot be accessed due to permissions
            return [
                {
                    "level": "ERROR",
                    "message": f"Permission denied accessing log file: {log_file_path}",
                    "raw_line": f"Permission error reading {log_file_path}",
                    "line_number": 0,
                }
            ]
        except Exception as e:
            # Return error information if file reading fails for other reasons
            return [
                {
                    "level": "ERROR",
                    "message": f"Failed to read log file: {e!s}",
                    "raw_line": f"Error reading {log_file_path}",
                    "line_number": 0,
                }
            ]

        return job_logs

    @classmethod
    def get_job_logs_summary(
        cls, job_id: str, log_filename: str = "job.log"
    ) -> dict[str, any]:
        """
        Get a summary of log entries for a specific job ID.

        Args:
            job_id: The job ID to search for
            log_filename: The log file to search in

        Returns:
            Dictionary containing:
            - total_logs: Total number of log entries
            - by_level: Count of logs by level (INFO, ERROR, etc.)
            - first_log: First log entry timestamp
            - last_log: Last log entry timestamp
            - has_errors: Boolean indicating if there are ERROR level logs
            - logs: List of all log entries
        """
        logs = cls.get_job_logs(job_id, log_filename)

        if not logs:
            return {
                "total_logs": 0,
                "by_level": {},
                "first_log": None,
                "last_log": None,
                "has_errors": False,
                "logs": [],
            }

        # Count logs by level
        by_level = {}
        for log in logs:
            level = log["level"]
            by_level[level] = by_level.get(level, 0) + 1

        # Get first and last timestamps
        first_log = logs[0].get("timestamp", "UNKNOWN")
        last_log = logs[-1].get("timestamp", "UNKNOWN")

        return {
            "total_logs": len(logs),
            "by_level": by_level,
            "first_log": first_log,
            "last_log": last_log,
            "has_errors": "ERROR" in by_level,
            "logs": logs,
        }
