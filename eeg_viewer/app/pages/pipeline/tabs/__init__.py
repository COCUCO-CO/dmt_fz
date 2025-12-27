"""
Pipeline tabs module.

Each tab follows the Component pattern:
- Self-contained rendering logic
- Minimal dependencies on parent
- Clear interface via render() method

Exports all tab classes.
"""

from .console import ConsoleTab
from .files import FilesTab
from .system import SystemTab
from .visualize import VisualizeTab


__all__ = [
    'ConsoleTab',
    'FilesTab',
    'SystemTab',
    'VisualizeTab',
]





