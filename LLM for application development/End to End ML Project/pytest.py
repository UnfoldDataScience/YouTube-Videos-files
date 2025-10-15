"""Lightweight pytest shim.

This module provides a minimal wrapper around Python's built‑in unittest
framework so that ``pytest -q`` can be used in environments where the
actual pytest library is unavailable.  It simply discovers and runs all
unittest tests in the ``tests/`` directory.  Extra command line arguments
are ignored except for ``-q``, which suppresses verbose output.
"""

from __future__ import annotations

import sys
import unittest


def main() -> None:
    # Determine verbosity: quiet if '-q' present
    quiet = "-q" in sys.argv
    loader = unittest.TestLoader()
    suite = loader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=0 if quiet else 2)
    result = runner.run(suite)
    # Exit with appropriate return code
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":  # pragma: no cover
    main()