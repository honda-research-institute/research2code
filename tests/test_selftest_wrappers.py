"""B-07 step 1: pytest wrappers for the orphan self-test suites nothing ran.

Same pattern as test_dispatch_hardening's dispatch_templates wrapper: the
suites keep executing INSIDE their own modules, so every self-patching
site (sys.modules[__name__] / globals() assignments) resolves in the
right namespace and zero rewrites are needed. Bodies migrate for real
under B-07 step 2/3; each migration deletes its wrapper here and
repoints the module's __main__ arm.
"""

from __future__ import annotations


def test_opencode_client_self_test_runs_in_ci():
    import opencode_client

    opencode_client._self_test()


def test_halt_catalog_self_test_runs_in_ci():
    import halt_catalog

    halt_catalog._self_test()
