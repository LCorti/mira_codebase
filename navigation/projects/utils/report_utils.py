import glob
import os
from datetime import datetime

from django.conf import settings


def get_reports_for_project(project_name=None):
    """
    Scan the REPORT_DIR for saved reports and organize them by project.

    Args:
        project_name (str, optional): If provided, only return reports for this specific project

    Returns:
        dict: Reports organized by project name with metadata
    """
    report_dir = getattr(settings, "REPORT_DIR", None)

    if (
        not report_dir
        or report_dir == "Not configured"
        or not os.path.exists(report_dir)
    ):
        return {}

    reports = {}

    try:
        if project_name:
            # Look for reports in the specific project directory
            project_dir = os.path.join(report_dir, project_name)
            if os.path.exists(project_dir):
                reports[project_name] = _scan_project_reports(project_dir, project_name)
        else:
            # Scan all project directories
            for item in os.listdir(report_dir):
                item_path = os.path.join(report_dir, item)
                if os.path.isdir(item_path):
                    project_reports = _scan_project_reports(item_path, item)
                    if project_reports:
                        reports[item] = project_reports
    except Exception as e:
        print(f"Error scanning reports directory: {e!s}")
        return {}

    return reports


def _scan_project_reports(project_dir, project_name):
    """
    Scan a specific project directory for report files.

    Args:
        project_dir (str): Path to the project's report directory
        project_name (str): Name of the project

    Returns:
        list: List of report metadata dictionaries
    """
    reports = []

    # Look for both HTML and PDF reports
    patterns = [os.path.join(project_dir, "*.html"), os.path.join(project_dir, "*.pdf")]

    for pattern in patterns:
        for file_path in glob.glob(pattern):
            try:
                file_stat = os.stat(file_path)
                file_name = os.path.basename(file_path)
                file_size = file_stat.st_size
                modified_time = datetime.fromtimestamp(file_stat.st_mtime)
                created_time = datetime.fromtimestamp(file_stat.st_ctime)

                # Extract metadata from filename if it follows the pattern
                # job_comparison_report_{project}_{method}_{timestamp}.{ext}
                report_info = _parse_report_filename(file_name)

                report_data = {
                    "id": _generate_report_id(file_path),
                    "name": file_name,
                    "display_name": report_info.get("display_name", file_name),
                    "file_path": file_path,
                    "relative_path": os.path.relpath(
                        file_path,
                        settings.REPORT_DIR if hasattr(settings, "REPORT_DIR") else "/",
                    ),
                    "file_type": os.path.splitext(file_name)[1].lower(),
                    "file_size": file_size,
                    "file_size_human": _format_file_size(file_size),
                    "created_at": created_time,
                    "modified_at": modified_time,
                    "last_edited": modified_time.strftime("%Y-%m-%d %H:%M"),
                    "project_name": project_name,
                    "method_name": report_info.get("method_name"),
                    "timestamp_str": report_info.get("timestamp_str"),
                }

                reports.append(report_data)

            except Exception as e:
                print(f"Error processing report file {file_path}: {e!s}")
                continue

    # Sort reports by modified time (newest first)
    reports.sort(key=lambda x: x["modified_at"], reverse=True)

    return reports


def _parse_report_filename(filename):
    """
    Parse report filename to extract metadata.
    Expected format: job_comparison_report_{project}_{method}_{timestamp}.{ext}

    Args:
        filename (str): The report filename

    Returns:
        dict: Parsed metadata
    """
    try:
        # Remove extension
        name_without_ext = os.path.splitext(filename)[0]

        # Check if it matches our expected pattern
        if name_without_ext.startswith("job_comparison_report_"):
            # Remove the prefix
            remaining = name_without_ext[22:]  # len('job_comparison_report_') = 22

            # Split by underscore and try to extract components
            parts = remaining.split("_")

            # The last part should be the timestamp (format: YYYYMMDD_HHMMSS)
            if len(parts) >= 3 and len(parts[-1]) == 6 and len(parts[-2]) == 8:
                timestamp_str = f"{parts[-2]}_{parts[-1]}"
                # Everything except the last 2 parts should be project_method
                project_method = "_".join(parts[:-2])

                # Try to split project and method (this is tricky since both can contain underscores)
                # We'll make a reasonable guess - assume method name is the last component before timestamp
                if "_" in project_method:
                    method_parts = project_method.split("_")
                    method_name = method_parts[-1]
                    project_name = "_".join(method_parts[:-1])
                else:
                    project_name = project_method
                    method_name = None

                display_name = (
                    f"{method_name} - {timestamp_str}" if method_name else filename
                )

                return {
                    "display_name": display_name,
                    "project_name": project_name,
                    "method_name": method_name,
                    "timestamp_str": timestamp_str,
                }
    except Exception as e:
        print(e)

    return {"display_name": filename}


def _generate_report_id(file_path):
    """
    Generate a unique ID for a report based on its file path.

    Args:
        file_path (str): Full path to the report file

    Returns:
        str: Unique identifier for the report
    """
    # Use the relative path from REPORT_DIR as the ID, replacing path separators
    report_dir = getattr(settings, "REPORT_DIR", "")
    if report_dir and file_path.startswith(report_dir):
        relative_path = os.path.relpath(file_path, report_dir)
        return relative_path.replace(os.path.sep, "_").replace(".", "_")
    else:
        # Fallback to filename-based ID
        return os.path.basename(file_path).replace(".", "_")


def _format_file_size(size_bytes):
    """
    Format file size in human-readable format.

    Args:
        size_bytes (int): File size in bytes

    Returns:
        str: Human-readable file size
    """
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    size = float(size_bytes)

    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1

    return f"{size:.1f} {size_names[i]}"


def get_report_content(report_id, project_name):
    """
    Get the content of a specific report file.

    Args:
        report_id (str): The report ID
        project_name (str): The project name

    Returns:
        tuple: (content, content_type) or (None, None) if not found
    """
    reports = get_reports_for_project(project_name)
    project_reports = reports.get(project_name, [])

    report = next((r for r in project_reports if r["id"] == report_id), None)

    if not report or not os.path.exists(report["file_path"]):
        return None, None

    try:
        file_type = report["file_type"]

        if file_type == ".html":
            with open(report["file_path"], "r", encoding="utf-8") as f:
                content = f.read()
            return content, "text/html"
        elif file_type == ".pdf":
            with open(report["file_path"], "rb") as f:
                content = f.read()
            return content, "application/pdf"
        else:
            return None, None

    except Exception as e:
        print(f"Error reading report file {report['file_path']}: {e!s}")
        return None, None
