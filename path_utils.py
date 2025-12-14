import os
import sys

def get_base_path():
    """Get the base path for resources, handling PyInstaller's _MEIPASS."""
    if hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

def get_resource_path(relative_path):
    """Get absolute path to a resource."""
    return os.path.join(get_base_path(), relative_path)
