#!/usr/bin/env python3
"""Sanity tests for the diversity metrics in metrics/ (vendi, mpcd, chamfer, mmd2),
using cases with analytically-known answers.

Reference values (cosine kernel; mpcd is a DISTANCE = 1 - mean cos):
  - N identical points     -> vendi=1,  mpcd=0            (minimum diversity)
  - D orthonormal vectors  -> vendi=D,  mpcd=1            (maximum diversity)
  - k orthogonal clusters  -> vendi=k
  - antipodal (+/-v)       -> vendi=1   (single direction)
  - chamfer / mmd2 of a set with itself -> 0; grow for a far / different set.
  - cosine metrics (vendi, mpcd) are scale-invariant per row.

Run:  python metrics/tests/test_diversity_metrics.py
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # metrics/
from vendi import vendi
from mpcd import mpcd
from chamfer import chamfer
from mmd import mmd2

D = 192
rng = np.random.default_rng(0)
_pass, _fail = [], []


def check(name, got, expect, tol):
    ok = abs(got - expect) < tol
    (_pass if ok else _fail).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name:46s} got={got:.4f} expect~={expect} (tol {tol})")


def main():
    # 1) identical points -> vendi 1, mpcd 0 (minimum diversity)
    X = np.tile(rng.standard_normal(D), (500, 1))
    check("identical x500: vendi==1", vendi(X), 1.0, 1e-6)
    check("identical x500: mpcd==0", mpcd(X), 0.0, 1e-6)
    for k in (2, 10, 1000):
        check(f"identical x{k}: vendi==1", vendi(np.tile(rng.standard_normal(D), (k, 1))), 1.0, 1e-6)

    # 2) orthonormal basis -> vendi D, mpcd 1 (maximum diversity)
    I = np.eye(D)
    check("orthonormal: vendi==D", vendi(I), D, 1e-3)
    check("orthonormal: mpcd==1", mpcd(I), 1.0, 1e-9)

    # 3) antipodal (+/-v): single direction -> vendi 1
    a = rng.standard_normal(D)
    check("antipodal 250/250: vendi==1",
          vendi(np.vstack([np.tile(a, (250, 1)), np.tile(-a, (250, 1))])), 1.0, 1e-6)

    # 4) two orthogonal clusters -> vendi 2; mpcd = 1 - mean cos (within=1, cross=0)
    X2 = np.vstack([np.tile(I[0], (250, 1)), np.tile(I[1], (250, 1))])
    check("2 orth clusters: vendi==2", vendi(X2), 2.0, 1e-6)
    check("2 orth clusters: mpcd", mpcd(X2), 1 - (250 * 249 * 2) / (500 * 499), 1e-3)

    # 5) cosine-metric scale invariance
    X = rng.standard_normal((300, D)); s = rng.uniform(0.1, 10, (300, 1))
    check("scale-invariance: vendi", vendi(X * s), vendi(X), 1e-6)
    check("scale-invariance: mpcd", mpcd(X * s), mpcd(X), 1e-6)

    # 6) random gaussian strictly inside (1, n) for vendi, (0,1) for mpcd
    X = rng.standard_normal((500, D)); vg = vendi(X); mg = mpcd(X)
    check("random: 1<vendi<n", 1.0 if 1.0 < vg < 500 else 0.0, 1.0, 0.5)
    check("random: 0<mpcd<1", 1.0 if 0.0 < mg < 1.0 else 0.0, 1.0, 0.5)

    # 7) cross-set distances: chamfer, mmd2
    A = rng.standard_normal((400, D))
    B = rng.standard_normal((400, D))          # same distribution, different sample
    far = rng.standard_normal((400, D)) + 50.0  # far-shifted cluster
    check("chamfer(X,X)==0", chamfer(A, A), 0.0, 1e-9)
    check("chamfer symmetric", chamfer(A, B), chamfer(B, A), 1e-9)
    check("chamfer(X,far)>>0", 1.0 if chamfer(A, far) > 10 else 0.0, 1.0, 0.5)
    check("mmd2(X,X)==0", mmd2(A, A), 0.0, 1e-9)
    check("mmd2 symmetric", mmd2(A, B), mmd2(B, A), 1e-9)
    check("mmd2(same-dist)~=0", mmd2(A, B), 0.0, 0.05)
    check("mmd2(X,far)>0", 1.0 if mmd2(A, far) > 0 else 0.0, 1.0, 0.5)

    print(f"\n=== {len(_pass)} passed, {len(_fail)} failed ===")
    if _fail:
        print("FAILED:", _fail)
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
