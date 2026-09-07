"""Compatibility entrypoint for the canonical hierarchical workflow application."""

from ci.workflows import generate


def workflow_index(*, check=False, write=False, root=None):
    """Keep the prior CLI while checking/generating the complete workflow outputs."""
    return generate(root=root, check=check, write=write)
