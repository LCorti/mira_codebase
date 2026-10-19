from django import template

register = template.Library()

# Import existing filters from comparison_tags
from .comparison_tags import get_item

# Re-register the existing filter so it can be used
register.filter('get_item', get_item)

@register.filter
def default_if_none(value, default):
    """Return default value if the value is None, otherwise return the value"""
    return default if value is None else value

@register.filter
def make_list(value):
    """Convert a string into a list of characters"""
    if isinstance(value, str):
        return list(value)
    return value