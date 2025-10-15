"""Test importing the Streamlit app using unittest."""

from __future__ import annotations

import importlib
import unittest


class TestAppImport(unittest.TestCase):
    """Ensure that the app module can be imported and reloaded."""

    def test_import_app(self) -> None:
        """Attempt to import the app; skip if streamlit is unavailable."""
        try:
            import app  # noqa: F401
            importlib.reload(app)
        except ModuleNotFoundError as exc:
            # Skip the test if streamlit is not installed
            self.skipTest(f"streamlit not installed: {exc}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()