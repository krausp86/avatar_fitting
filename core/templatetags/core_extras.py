from django import template

register = template.Library()


@register.filter
def subtract(value, arg):
    """{{ value|subtract:arg }} — subtrahiert arg von value."""
    try:
        return int(value) - int(arg)
    except (TypeError, ValueError):
        return 0
