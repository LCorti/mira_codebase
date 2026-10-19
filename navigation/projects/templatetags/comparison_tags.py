import re
from datetime import datetime

import matplotlib.colors as mcolors
from django import template
from matplotlib import colormaps

register = template.Library()


@register.filter
def get_item(list_or_dict, key):
    """Get an item from a list or dictionary by key/index"""
    try:
        if isinstance(list_or_dict, list):
            return list_or_dict[int(key)]
        elif isinstance(list_or_dict, dict):
            return list_or_dict.get(key)
        else:
            return None
    except (ValueError, IndexError, TypeError):
        return None


@register.filter
def get_sample_result(job, sample_index):
    """Get a specific sample result from a job"""
    try:
        if job.get("result") and job["result"].get("results"):
            results = job["result"]["results"]
            if sample_index < len(results):
                return results[sample_index]
        return None
    except (TypeError, KeyError, IndexError):
        return None


@register.filter
def get_sample_input(jobs, sample_index):
    """Get the input for a specific sample (from the first job with results)"""
    try:
        for job in jobs:
            if job.get("result") and job["result"].get("results"):
                results = job["result"]["results"]
                if sample_index < len(results):
                    return results[sample_index].get("input", "")
        return None
    except (TypeError, KeyError, IndexError):
        return None


@register.simple_tag
def get_first_available_documents(jobs, sample_index):
    """
    Merge and deduplicate documents from all jobs for the given sample.
    Returns a union of all documents, keeping the highest scores for duplicates.
    Returns a dict with 'documents', 'total_jobs', 'has_overlap', and 'source_job_ids' keys.
    Each document includes its reference value (relevance label).
    """
    try:
        # Dictionary to track documents by doc_id
        doc_map = {}
        source_jobs = set()

        for idx, job in enumerate(jobs):
            sample_result = get_sample_result(job, sample_index)
            if sample_result and sample_result.get("documents"):
                source_jobs.add(job.get("id", ""))
                documents = sample_result["documents"]
                references = sample_result.get("reference", [])

                for doc_idx, doc in enumerate(documents):
                    doc_id = doc.get("doc_id")
                    if doc_id is None:
                        continue

                    # Get reference value for this document at its position
                    reference_value = (
                        references[doc_idx] if doc_idx < len(references) else None
                    )

                    # If this doc_id is new, add it
                    if doc_id not in doc_map:
                        doc_map[doc_id] = {
                            "doc_id": doc_id,
                            "doc_text": doc.get("doc_text", ""),
                            "bm25_score": doc.get("bm25_score"),
                            "rerank_score": doc.get("rerank_score"),
                            "reference": reference_value,
                            "source_jobs": [job.get("id", "")[:8]],
                            "job_count": 1,
                            "reference_values": [reference_value]
                            if reference_value is not None
                            else [],
                        }
                    else:
                        # Document exists - update with highest scores
                        existing = doc_map[doc_id]
                        existing["job_count"] += 1
                        existing["source_jobs"].append(job.get("id", "")[:8])

                        # Keep highest bm25_score
                        if doc.get("bm25_score") is not None and (
                            existing["bm25_score"] is None
                            or doc["bm25_score"] > existing["bm25_score"]
                        ):
                            existing["bm25_score"] = doc["bm25_score"]

                        # Keep highest rerank_score
                        if doc.get("rerank_score") is not None and (
                            existing["rerank_score"] is None
                            or doc["rerank_score"] > existing["rerank_score"]
                        ):
                            existing["rerank_score"] = doc["rerank_score"]

                        # Collect all reference values for this document
                        if reference_value is not None:
                            existing["reference_values"].append(reference_value)
                            # Use any non-None reference value (they should all be the same for the same document)
                            if existing["reference"] is None:
                                existing["reference"] = reference_value

        if not doc_map:
            return None

        # Convert to list and sort by relevance (rerank_score > bm25_score)
        documents = list(doc_map.values())
        documents.sort(
            key=lambda d: (
                d["rerank_score"] if d["rerank_score"] is not None else -1,
                d["bm25_score"] if d["bm25_score"] is not None else -1,
            ),
            reverse=True,
        )

        # Check if there's any overlap
        has_overlap = any(doc["job_count"] > 1 for doc in documents)

        return {
            "documents": documents,
            "total_jobs": len(source_jobs),
            "has_overlap": has_overlap,
            "source_job_ids": list(source_jobs),
        }
    except (TypeError, KeyError, IndexError) as e:
        # Log error for debugging
        print(f"Error in get_first_available_documents: {e}")
        return None


@register.filter
def get_doc_id_at_index(sample_result, index):
    """
    Get the document ID at a specific index from a sample result's documents.
    Returns the doc_id if available, otherwise returns the index number.
    """
    try:
        if sample_result and sample_result.get("documents"):
            documents = sample_result["documents"]
            if index < len(documents):
                return documents[index].get("doc_id", index)
        return index
    except (TypeError, KeyError, IndexError, AttributeError):
        return index


@register.filter
def truncate_text(text, length=200):
    """Truncate text to specified length"""
    if not text:
        return ""
    text_str = str(text)
    if len(text_str) <= length:
        return text_str
    return text_str[:length] + "..."


@register.filter
def format_score(value, decimal_places=4):
    """Format a numeric score to specified decimal places"""
    try:
        if value is None:
            return "—"
        return f"{float(value):.{decimal_places}f}"
    except (ValueError, TypeError):
        return str(value) if value is not None else "—"


@register.filter
def format_metric_list(value, decimal_places=2):
    """
    Format a metric value that can be either a single value or a list.
    - If single value: format it
    - If list with 1 element: format that element
    - If list with multiple elements: format each element separated by commas
    """
    try:
        if value is None:
            return "—"

        # Check if it's a list or tuple
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return "—"
            elif len(value) == 1:
                # Single element list - format it
                return f"{float(value[0]):.{decimal_places}f}"
            else:
                # Multiple elements - format each
                formatted_values = [
                    f"{float(v):.{decimal_places}f}" for v in value if v is not None
                ]
                return ", ".join(formatted_values)
        else:
            # Single value
            return f"{float(value):.{decimal_places}f}"
    except (ValueError, TypeError):
        return str(value) if value is not None else "—"


@register.filter
def get_metric_class(value, good_threshold=0.8, medium_threshold=0.5):
    """Get CSS class for metric value based on thresholds"""
    try:
        if value is None:
            return "metric-poor"
        float_value = float(value)
        if float_value >= good_threshold:
            return "metric-good"
        elif float_value >= medium_threshold:
            return "metric-medium"
        else:
            return "metric-poor"
    except (ValueError, TypeError):
        return "metric-poor"


@register.filter
def slice_documents(documents, count=6):
    """Get first N documents from a list"""
    try:
        if documents and isinstance(documents, list):
            return documents[:count]
        return []
    except (TypeError, AttributeError):
        return []


@register.filter
def slice_scores(scores, count=8):
    """Get first N scores from a list"""
    try:
        if scores and isinstance(scores, list):
            return scores[:count]
        return []
    except (TypeError, AttributeError):
        return []


@register.filter
def dict_items(dictionary):
    """Get items from a dictionary for iteration"""
    try:
        if isinstance(dictionary, dict):
            return dictionary.items()
        return []
    except (TypeError, AttributeError):
        return []


@register.filter
def is_relevant(value):
    """Check if a relevance value indicates relevance (1)"""
    try:
        return int(value) != 0
    except (ValueError, TypeError):
        return False


@register.filter
def get_score_class(value):
    """Get CSS class for score badges based on value"""
    try:
        if value is None:
            return "score-poor"

        float_value = float(value)
        if float_value >= 0.8:
            return "score-excellent"
        elif float_value >= 0.6:
            return "score-good"
        elif float_value >= 0.4:
            return "score-fair"
        else:
            return "score-poor"
    except (ValueError, TypeError):
        return "score-poor"


@register.filter
def slice(value, arg):
    """Slice a list or string - Django's slice filter equivalent"""
    try:
        if ":" in arg:
            start, end = arg.split(":")
            start = int(start) if start else None
            end = int(end) if end else None
            return value[start:end]
        else:
            return value[: int(arg)]
    except (ValueError, TypeError, AttributeError):
        return value


# Dynamic Color Utilities for Thresholds


def generate_threshold_colors(thresholds, colormap_name="RdYlGn"):
    """
    Generate colors for given thresholds using a matplotlib colormap.

    Args:
        thresholds (list): List of threshold values, sorted from high to low
        colormap_name (str): Name of the matplotlib colormap to use

    Returns:
        list: List of hex color codes corresponding to each threshold
    """
    if not thresholds:
        return []

    # Get the colormap
    cmap = colormaps[colormap_name]

    # Generate evenly spaced values across the colormap
    num_thresholds = len(thresholds)
    positions = [i / (num_thresholds - 1) for i in range(num_thresholds)]
    positions.reverse()  # Highest threshold gets green (position closer to 1)

    # Generate colors
    colors = []
    for pos in positions:
        rgba_color = cmap(pos)
        hex_color = mcolors.rgb2hex(rgba_color[:3])
        colors.append(hex_color)

    return colors


@register.filter
def dynamic_metric_color(value, thresholds_str="0.8,0.6,0.4"):
    """
    Get dynamic color for metric value based on custom thresholds.

    Args:
        value: The metric value to color
        thresholds_str: Comma-separated string of thresholds (high to low) Default: "0.8,0.6,0.4" for good, medium, poor

    Returns:
        str: Hex color code

    Usage in template:
        {{ metric_value|dynamic_metric_color:"0.9,0.7,0.5,0.3" }}
    """
    try:
        if value is None:
            return "#808080"  # Gray for None values

        # Parse thresholds
        thresholds = [float(t.strip()) for t in thresholds_str.split(",")]
        thresholds.sort(reverse=True)  # Ensure high to low order

        # Generate colors
        colors = generate_threshold_colors(thresholds)

        # Find appropriate color
        float_value = float(value)
        for i, threshold in enumerate(thresholds):
            if float_value >= threshold:
                return colors[i]

        # Below all thresholds - return worst color
        return colors[-1] if colors else "#808080"

    except (ValueError, TypeError, IndexError):
        return "#808080"  # Default gray


@register.filter
def dynamic_metric_class(value, thresholds_str="0.8,0.6,0.4"):
    """
    Get dynamic CSS class for metric value based on custom thresholds.

    Args:
        value: The metric value
        thresholds_str: Comma-separated string of thresholds (high to low)

    Returns:
        str: CSS class name

    Usage in template:
        {{ metric_value|dynamic_metric_class:"0.9,0.7,0.5,0.3" }}
    """
    try:
        if value is None:
            return "metric-poor"

        # Parse thresholds
        thresholds = [float(t.strip()) for t in thresholds_str.split(",")]
        thresholds.sort(reverse=True)  # Ensure high to low order

        # Define CSS classes based on number of thresholds
        if len(thresholds) == 2:
            css_classes = ["metric-good", "metric-poor"]
        elif len(thresholds) == 3:
            css_classes = ["metric-good", "metric-medium", "metric-poor"]
        elif len(thresholds) == 4:
            css_classes = [
                "metric-excellent",
                "metric-good",
                "metric-medium",
                "metric-poor",
            ]
        elif len(thresholds) >= 5:
            css_classes = [
                "metric-excellent",
                "metric-very-good",
                "metric-good",
                "metric-medium",
                "metric-poor",
            ]
        else:
            css_classes = ["metric-poor"]

        # Ensure we have enough classes for all thresholds
        while len(css_classes) < len(thresholds):
            css_classes.append("metric-poor")

        # Find appropriate class
        float_value = float(value)
        for i, threshold in enumerate(thresholds):
            if float_value >= threshold:
                return css_classes[i]

        # Below all thresholds
        return css_classes[-1] if css_classes else "metric-poor"

    except (ValueError, TypeError, IndexError):
        return "metric-poor"


@register.simple_tag
def generate_color_scale(thresholds_str="0.8,0.6,0.4", colormap="RdYlGn"):
    """
    Generate a color scale for legend/documentation purposes.

    Args:
        thresholds_str: Comma-separated string of thresholds
        colormap: Matplotlib colormap name

    Returns:
        list: List of dicts with threshold and color info

    Usage in template:
        {% generate_color_scale "0.9,0.7,0.5,0.3" "RdYlGn" as color_scale %}
    """
    try:
        thresholds = [float(t.strip()) for t in thresholds_str.split(",")]
        thresholds.sort(reverse=True)
        colors = generate_threshold_colors(thresholds, colormap)

        scale = []
        for i, (threshold, color) in enumerate(zip(thresholds, colors)):
            scale.append(
                {
                    "threshold": threshold,
                    "color": color,
                    "label": f">= {threshold}",
                    "index": i,
                }
            )

        return scale

    except (ValueError, TypeError):
        return []


@register.filter
def format_datetime(datetime_str, format_type="full"):
    """
    Format datetime string to a more readable format.

    Args:
        datetime_str: ISO datetime string like "2025-08-29T13:48:14.075704+00:00"
        format_type: "full" for "Aug 29, 2025 13:48", "short" for "Aug 29 13:48"

    Returns:
        Formatted datetime string or original if parsing fails
    """
    if not datetime_str:
        return "—"

    try:
        # Parse ISO format datetime string
        # Handle both with and without microseconds
        datetime_str = str(datetime_str)

        # Remove microseconds if present for easier parsing
        if "." in datetime_str and "+" in datetime_str:
            # Format: 2025-08-29T13:48:14.075704+00:00
            date_part, tz_part = datetime_str.split("+")
            date_part = date_part.split(".")[0]  # Remove microseconds
            datetime_str = date_part + "+" + tz_part
        elif "." in datetime_str and "Z" in datetime_str:
            # Format: 2025-08-29T13:48:14.075704Z
            datetime_str = datetime_str.split(".")[0] + "Z"

        # Try different datetime formats
        for fmt in ["%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"]:
            try:
                if datetime_str.endswith("Z"):
                    # Replace Z with +00:00 for parsing
                    datetime_str = datetime_str[:-1] + "+00:00"

                dt = datetime.strptime(datetime_str, fmt)

                # Format based on type
                if format_type == "short":
                    return dt.strftime("%b %d %H:%M")
                else:  # full
                    return dt.strftime("%b %d, %Y %H:%M")

            except ValueError:
                continue

        # If all parsing fails, try to extract date/time with regex
        match = re.match(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})", str(datetime_str))
        if match:
            year, month, day, hour, minute = match.groups()
            dt = datetime(int(year), int(month), int(day), int(hour), int(minute))
            if format_type == "short":
                return dt.strftime("%b %d %H:%M")
            else:
                return dt.strftime("%b %d, %Y %H:%M")

        # If everything fails, return the original string
        return str(datetime_str)

    except Exception:
        return str(datetime_str) if datetime_str else "—"


@register.filter
def truncatewords(text, num_words):
    """Truncate text to specified number of words"""
    if not text:
        return ""

    words = str(text).split()
    if len(words) <= num_words:
        return text

    truncated = " ".join(words[:num_words])
    return f"{truncated}..."


@register.filter
def wordcount(text):
    """Count the number of words in text"""
    if not text:
        return 0
    return len(str(text).split())


@register.filter
def escapejs(text):
    """Escape text for use in JavaScript"""
    if not text:
        return ""

    # Replace special characters that could break JavaScript
    text = str(text)
    text = text.replace("\\", "\\\\")  # Escape backslashes first
    text = text.replace("'", "\\'")  # Escape single quotes
    text = text.replace('"', '\\"')  # Escape double quotes
    text = text.replace("\n", "\\n")  # Escape newlines
    text = text.replace("\r", "\\r")  # Escape carriage returns
    text = text.replace("\t", "\\t")  # Escape tabs
    return text


@register.filter
def normalize_metric_value(value):
    """
    Normalize metric value to handle both single float and list formats.

    Args:
        value: Either a float/int or a list containing float/int values

    Returns:
        float: The normalized metric value (first value if list, or the value itself)

    Usage in template:
        {{ metric_value|normalize_metric_value|floatformat:3 }}
        {{ metric_value|normalize_metric_value|dynamic_metric_color:metric_thresholds }}
    """
    try:
        # Handle None values
        if value is None:
            return None

        # Handle list format - use first value
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return float(value[0]) if value[0] is not None else None

        # Handle direct float/int format
        if isinstance(value, (int, float)):
            return float(value)

        # Try to convert string to float
        if isinstance(value, str):
            return float(value)

        # Fallback for unexpected types
        return None

    except (ValueError, TypeError, IndexError):
        return None


@register.filter
def normalize_metric_for_display(value, metric_name):
    """
    Normalize metric value to 0-1 scale for consistent display across different metric types.
    This matches the JavaScript normalization logic used in charts.

    Args:
        value: The metric value (float, int, or list)
        metric_name: The name of the metric for determining the appropriate scaling

    Returns:
        float: The normalized metric value (0-1 scale) or None if invalid
    """
    try:
        # First normalize the value format (handle lists, etc.)
        normalized_value = normalize_metric_value(value)
        if normalized_value is None:
            return None

        # Define metric ranges - same as JavaScript version
        metric_ranges = {
            # Readability metrics (0-100 scale)
            "flesch_kincaid": (0, 100),
            "flesch": (0, 100),
            "flesch_reading_ease": (0, 100),
            "gunning_fog": (0, 20),
            "coleman_liau": (0, 20),
            "dale_chall": (0, 10),
            "ari": (0, 20),
            "linsear_write": (0, 20),
            "smog": (0, 20),
            "spache": (0, 10),
            # Text simplification/generation metrics
            "sari": (0, 100),
            # Rank metrics (higher rank = worse, but we'll handle separately)
            "median_rank_correct_word": (0, 50000),  # Typical vocabulary size
            # HARIM+ can be negative (typically -10 to 10)
            "harim_": (-10, 10),
            "harim_plus": (-10, 10),
            # Perplexity (lower is better, typical range 0-1000)
            "perplexity": (0, 1000),  # Typical good models: 10-100, acceptable: 100-500
        }

        # Normalize metric name (convert to lowercase, replace non-alphanumeric with underscore)
        normalized_name = re.sub(r"[^a-z_]", "_", str(metric_name).lower())

        # Get range for this metric (default to 0-1 if not found)
        min_val, max_val = metric_ranges.get(normalized_name, (0, 1))

        # Metrics where lower values are better
        lower_is_better = ["median_rank_correct_word", "perplexity"]

        if normalized_name in lower_is_better:
            # For "lower is better" metrics, invert the scale
            clamped_value = max(min_val, min(max_val, normalized_value))
            return 1 - ((clamped_value - min_val) / (max_val - min_val))
        elif normalized_name in [
            "flesch_kincaid",
            "flesch",
            "flesch_reading_ease",
            "sari",
        ]:
            # Metrics on 0-100 scale where higher is better
            return max(0, min(100, normalized_value)) / 100
        elif normalized_name in ["harim_", "harim_plus"]:
            # HARIM+ can be negative; normalize to 0-1 where higher (closer to 10) is better
            clamped_value = max(min_val, min(max_val, normalized_value))
            return (clamped_value - min_val) / (max_val - min_val)
        else:
            # For metrics already on 0-1 scale or where higher is better
            clamped_value = max(min_val, min(max_val, normalized_value))
            return (clamped_value - min_val) / (max_val - min_val)

    except (ValueError, TypeError, AttributeError):
        return None


def generate_lxt_token_colors(scores, colormap_name="RdBu_r"):
    """
    Generate colors for LXT token scores using a matplotlib colormap with magnitude-based scaling.
    Values close to 0 are nearly colorless, values close to -1/+1 are highly saturated.

    Args:
        scores (list): List of score values
        colormap_name (str): Name of the matplotlib colormap to use ('RdBu_r' for red-blue reversed)

    Returns:
        dict: Dictionary mapping scores to hex color codes
    """
    if not scores:
        return {}

    # Get the colormap
    cmap = colormaps[colormap_name]

    # Generate colors for each score using magnitude-based scaling
    color_map = {}
    for score in scores:
        magnitude = abs(score)
        is_positive = score > 0

        # Clamp magnitude to [0, 1] range
        clamped_magnitude = max(0, min(1, magnitude))

        if colormap_name == "RdBu_r":
            # For RdBu_r: 0 = red (negative), 0.5 = white (neutral), 1 = blue (positive)
            # Map based on magnitude and sign
            if is_positive:
                # Positive values: interpolate from neutral (0.5) to blue (1.0)
                t = 0.5 + (clamped_magnitude * 0.5)
            else:
                # Negative values: interpolate from neutral (0.5) to red (0.0)
                t = 0.5 - (clamped_magnitude * 0.5)
        else:
            # For other colormaps, use magnitude directly
            t = clamped_magnitude

        # Get color from colormap
        rgba_color = cmap(t)
        hex_color = mcolors.rgb2hex(rgba_color[:3])
        color_map[score] = hex_color

    return color_map


@register.simple_tag
def generate_lxt_colors(lxt_data, colormap="RdBu_r"):
    """
    Generate color mapping for LXT token data.

    Args:
        lxt_data: LXT diagnostic data (array with single object containing token-score pairs)
        colormap: Matplotlib colormap name (default: 'RdBu_r' for red-negative, blue-positive)

    Returns:
        dict: Color mapping for JavaScript consumption

    Usage in template:
        {% generate_lxt_colors job.result.diagnostics.lxt "RdBu_r" as lxt_colors %}
    """
    try:
        if not isinstance(lxt_data, list) or len(lxt_data) == 0:
            return {}

        # Extract the token object (first element of the array)
        token_object = lxt_data[0]
        if not isinstance(token_object, dict):
            return {}

        # Extract all scores
        scores = [
            score for score in token_object.values() if isinstance(score, (int, float))
        ]

        if not scores:
            return {}

        # Generate color mapping
        color_map = generate_lxt_token_colors(scores, colormap)

        # Return data structure for JavaScript
        return {
            "colors": color_map,
            "tokens": token_object,
            "min_score": min(scores),
            "max_score": max(scores),
            "colormap": colormap,
        }

    except (ValueError, TypeError, AttributeError):
        return {}


@register.filter
def lxt_token_color(score, color_data):
    """
    Get color for a specific LXT token score.

    Args:
        score: The token's relevance score
        color_data: Color data generated by generate_lxt_colors

    Returns:
        str: Hex color code

    Usage in template:
        {{ token_score|lxt_token_color:lxt_colors }}
    """
    try:
        if not isinstance(color_data, dict) or "colors" not in color_data:
            return "#808080"  # Default gray

        return color_data["colors"].get(score, "#808080")

    except (ValueError, TypeError, AttributeError):
        return "#808080"


@register.filter
def short_id(job_id):
    """Get a short version of a job ID (first 8 characters)"""
    try:
        if job_id:
            return str(job_id)[:8]
        return ""
    except (TypeError, AttributeError):
        return ""


@register.filter
def jobs_have_tags(jobs):
    """Check if any of the jobs have tags"""
    try:
        if not jobs:
            return False
        for job in jobs:
            if job.get("tags") and len(job["tags"]) > 0:
                return True
        return False
    except (TypeError, AttributeError):
        return False


@register.filter
def jobs_have_diagnostics(jobs):
    """Check if any of the jobs have diagnostics"""
    try:
        if not jobs:
            return False
        for job in jobs:
            if (
                job.get("result")
                and job["result"].get("diagnostics")
                and len(job["result"]["diagnostics"]) > 0
            ):
                return True
        return False
    except (TypeError, AttributeError):
        return False


@register.filter
def jobs_have_shap(jobs):
    """Check if any of the jobs have SHAP diagnostics"""
    try:
        if not jobs:
            return False
        for job in jobs:
            if (
                job.get("result")
                and job["result"].get("diagnostics")
                and job["result"]["diagnostics"].get("shap")
            ):
                return True
        return False
    except (TypeError, AttributeError):
        return False


@register.filter
def jobs_have_lxt(jobs):
    """Check if any of the jobs have LXT diagnostics"""
    try:
        if not jobs:
            return False
        for job in jobs:
            if (
                job.get("result")
                and job["result"].get("diagnostics")
                and job["result"]["diagnostics"].get("lxt")
            ):
                return True
        return False
    except (TypeError, AttributeError):
        return False


@register.filter
def jobs_have_shap_or_lxt(jobs):
    """Check if any of the jobs have SHAP or LXT diagnostics"""
    return jobs_have_shap(jobs) or jobs_have_lxt(jobs)


@register.filter
def jobs_have_shap_encoder_or_ig(jobs):
    return jobs_have_shap_encoder(jobs) or jobs_have_ig(jobs)


@register.filter
def jobs_have_shap_encoder(jobs):
    """Check if any of the jobs have SHAP Encoder diagnostics"""
    try:
        if not jobs:
            return False
        for job in jobs:
            if (
                job.get("result")
                and job["result"].get("diagnostics")
                and job["result"]["diagnostics"].get("shap_encoder")
            ):
                return True
        return False
    except (TypeError, AttributeError):
        return False


@register.filter
def jobs_have_ig(jobs):
    """Check if any of the jobs have Integrated Gradients diagnostics"""
    try:
        if not jobs:
            return False
        for job in jobs:
            if (
                job.get("result")
                and job["result"].get("diagnostics")
                and job["result"]["diagnostics"].get("integrated_gradients_encoder")
            ):
                return True
        return False
    except (TypeError, AttributeError):
        return False


@register.filter
def any_jobs_have_diagnostics(jobs):
    """Check if any jobs have any diagnostics (including SHAP/LXT or other diagnostics)"""
    try:
        if not jobs:
            return False
        for job in jobs:
            if (
                job.get("result")
                and job["result"].get("diagnostics")
                and len(job["result"]["diagnostics"]) > 0
            ):
                return True
        return False
    except (TypeError, AttributeError):
        return False


@register.filter
def get_type(value):
    """Get the type of a value as a string"""
    try:
        if isinstance(value, list):
            return "list"
        elif isinstance(value, dict):
            return "dict"
        elif isinstance(value, str):
            return "str"
        else:
            return type(value).__name__
    except Exception:
        return "unknown"


@register.filter
def lxt_token_color(score):
    """Get color for an LXT token score using a simple red-blue scale"""
    try:
        # Convert score to float
        score_val = float(score)

        # Clamp to [-1, 1] range
        clamped_score = max(-1, min(1, score_val))

        # Map to color: negative scores -> red, positive scores -> blue, zero -> neutral
        if clamped_score > 0:
            # Positive: blue scale (light blue to dark blue)
            intensity = min(255, int(255 * abs(clamped_score)))
            return f"rgb({255 - intensity}, {255 - intensity}, 255)"
        elif clamped_score < 0:
            # Negative: red scale (light red to dark red)
            intensity = min(255, int(255 * abs(clamped_score)))
            return f"rgb(255, {255 - intensity}, {255 - intensity})"
        else:
            # Zero: neutral gray
            return "rgb(240, 240, 240)"

    except (ValueError, TypeError):
        return "rgb(240, 240, 240)"  # Default neutral


@register.filter
def lxt_text_color(score):
    """Get appropriate text color for an LXT token based on background"""
    try:
        score_val = float(score)
        clamped_score = max(-1, min(1, score_val))

        # Use dark text for light backgrounds, light text for dark backgrounds
        if abs(clamped_score) > 0.6:
            return "white"
        else:
            return "black"

    except (ValueError, TypeError):
        return "black"


@register.filter
def format_datetime(value, format_type="full"):
    """Format datetime value with different formats"""
    try:
        if not value:
            return ""

        # If value is a string, try to parse it
        if isinstance(value, str):
            # Try parsing ISO format
            try:
                value = datetime.fromisoformat(value)
            except ValueError:
                return str(value)

        if format_type == "full":
            return value.strftime("%B %d, %Y at %H:%M")
        elif format_type == "short":
            return value.strftime("%m/%d %H:%M")
        else:
            return value.strftime("%Y-%m-%d %H:%M:%S")

    except (ValueError, TypeError, AttributeError):
        return str(value) if value else ""


@register.filter
def get_source_metric(metric_stats):
    """Get the source_metric attribute from metric stats if available"""
    try:
        if isinstance(metric_stats, dict):
            return metric_stats.get("source_metric", None)
        return None
    except (TypeError, AttributeError):
        return None


@register.simple_tag
def get_source_metric_from_job(job, metric_name):
    """Get the source_metric for a given metric_name from a job's summary"""
    try:
        if (
            job.get("result")
            and job["result"].get("summary")
            and metric_name in job["result"]["summary"]
        ):
            metric_stats = job["result"]["summary"][metric_name]
            if isinstance(metric_stats, dict):
                return metric_stats.get("source_metric", None)
        return None
    except (TypeError, KeyError, AttributeError):
        return None
