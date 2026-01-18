"""Filter modules for MCP tool results."""

from .base import BaseResultFilter
from .cochrane import CochraneResultFilter, create_title_filter_from_list

__all__ = [
    "BaseResultFilter",
    "CochraneResultFilter",
    "create_title_filter_from_list",
]

