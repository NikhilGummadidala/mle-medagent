"""Specialist models. Each module exposes ``predict(...)`` returning a dict with
``diagnosis``, ``confidence`` (float or None), ``summary`` and ``details``.

Heavy frameworks are imported lazily inside each module, so importing this
package is cheap.
"""
