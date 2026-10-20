# Import all views here to make them available when importing from this package
from .method import edit
from .project import create, delete, edit, open, show
from .test import create, edit

method_funcs = ["edit"]
project_funcs = ["create", "delete", "edit", "open", "show"]
test_funcs = ["create"]

__all__ = project_funcs + test_funcs + method_funcs  # noqa
