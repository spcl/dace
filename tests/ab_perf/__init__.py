"""A/B performance comparison tests for canonicalize / transformation knobs.

These tests are opt-in (the default test sweep skips them) and time CPU and
GPU variants of a single kernel under two transformation settings. See
``conftest.py`` for the opt-in flag and shared options; see ``_harness.py``
for the timing helpers and CPU/GPU device dispatch.
"""
