"""Sprint 14 (DOE Phase 3 alignment, C3): deterministic seed derivation.

Python's builtin `hash()` on str/tuple-of-str is salted per-process by
default (PEP 456, hash-flooding protection) unless `PYTHONHASHSEED` is fixed
in the environment. Every experiment script since Sprint 2 that derived a
per-instance seed via `hash((name, tag, i)) % (2**31)` therefore drew a
*different* random instance on every separate `python script.py` invocation,
not a reproducible one -- silently, since each individual run still produces
internally-consistent, statistically valid results (the bug only breaks
claims of the form "these two separate runs used the identical instances").
Discovered in Sprint 14 while investigating why a contingency-demo script's
"same seed formula" instance search kept turning up an all-trivial-optimum
set; confirmed directly (two `python -c` calls hashing the identical tuple
in separate processes returned different integers).

`stable_seed(*parts)` replaces the `hash(...)` idiom with a real
process-independent hash (Python's `zlib.crc32` over a canonical string
join), so `stable_seed(name, tag, i)` returns the identical integer every
time, in every process, forever -- the actual reproducibility guarantee the
DOE Phase 3 rubric's #1 named gap requires ("solutions must be reproducible
... sufficient for a third-party reviewer to verify the headline results").
"""
import zlib


def stable_seed(*parts) -> int:
    key = "|".join(str(p) for p in parts)
    return zlib.crc32(key.encode("utf-8"))
