import matplotlib.colors as mcolors
from matplotlib import colormaps


def generate_threshold_colors(thresholds, colormap_name="RdYlGn"):
    """
    Generate colors for given thresholds using a matplotlib colormap.

    Args:
        thresholds (list): List of threshold values, sorted from high to low e.g., [0.9, 0.7, 0.5, 0.3] for excellent, good, fair, poor
        colormap_name (str): Name of the matplotlib colormap to use (default: 'RdYlGn')

    Returns:
        list: List of hex color codes corresponding to each threshold

    Example:
        >>> thresholds = [0.9, 0.7, 0.5, 0.3]
        >>> colors = generate_threshold_colors(thresholds)
        >>> print(colors)
        ['#006837', '#31a354', '#fed976', '#d73027']
    """
    if not thresholds:
        return []

    # Get the colormap
    cmap = colormaps[colormap_name]

    # Generate evenly spaced values across the colormap
    # For RdYlGn: 0 = red (bad), 1 = green (good)
    # We want highest threshold to get green (good), lowest to get red (bad)
    num_thresholds = len(thresholds)

    # Create normalized positions for each threshold
    # Reverse the order so highest threshold gets green (position closer to 1)
    positions = [i / (num_thresholds - 1) for i in range(num_thresholds)]
    positions.reverse()  # Now highest threshold gets position closest to 1 (green)

    # Generate colors
    colors = []
    for pos in positions:
        # Get RGBA color from colormap
        rgba_color = cmap(pos)
        # Convert to hex
        hex_color = mcolors.rgb2hex(rgba_color[:3])
        colors.append(hex_color)

    return colors


def get_color_for_value(value, thresholds, colors):
    """
    Get the appropriate color for a given value based on thresholds.

    Args:
        value (float): The value to get color for
        thresholds (list): List of threshold values, sorted from high to low
        colors (list): List of corresponding colors for each threshold

    Returns:
        str: Hex color code for the value

    Example:
        >>> thresholds = [0.9, 0.7, 0.5, 0.3]
        >>> colors = generate_threshold_colors(thresholds)
        >>> color = get_color_for_value(0.8, thresholds, colors)
        >>> print(color)  # Will return color for 0.7 threshold (good)
    """
    if not thresholds or not colors or len(thresholds) != len(colors):
        return "#808080"  # Default gray

    try:
        value = float(value)
    except (ValueError, TypeError):
        return "#808080"  # Default gray for invalid values

    # Find the appropriate threshold
    for i, threshold in enumerate(thresholds):
        if value >= threshold:
            return colors[i]

    # If value is below all thresholds, return the last (worst) color
    return colors[-1]


def get_css_class_for_value(value, thresholds, css_classes=None):
    """
    Get the appropriate CSS class for a given value based on thresholds.

    Args:
        value (float): The value to get CSS class for
        thresholds (list): List of threshold values, sorted from high to low
        css_classes (list): List of CSS class names. If None, defaults to ['excellent', 'good', 'fair', 'poor']

    Returns:
        str: CSS class name for the value
    """
    if css_classes is None:
        # Default CSS classes for up to 4 thresholds
        default_classes = ["excellent", "good", "fair", "poor"]
        css_classes = default_classes[: len(thresholds)] if thresholds else ["poor"]

    if not thresholds or not css_classes or len(thresholds) != len(css_classes):
        return "poor"  # Default class

    try:
        value = float(value)
    except (ValueError, TypeError):
        return "poor"  # Default class for invalid values

    # Find the appropriate threshold
    for i, threshold in enumerate(thresholds):
        if value >= threshold:
            return css_classes[i]

    # If value is below all thresholds, return the last (worst) class
    return css_classes[-1]
