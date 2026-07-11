"""DEPRECATED wrapper — use validate_paper_evidence.py.

Kept only so stale invocations fail loudly toward the canonical validator
instead of silently checking a stale contract. It delegates directly to
validate_paper_evidence.main() and forwards its exit status.
"""
import sys
import warnings

from validate_paper_evidence import main

if __name__ == "__main__":
    warnings.warn(
        "validate_results.py is deprecated; run "
        "'python src/validate_paper_evidence.py' instead.",
        DeprecationWarning, stacklevel=1)
    print("[validate_results] DEPRECATED: delegating to "
          "validate_paper_evidence.py")
    sys.exit(main())
