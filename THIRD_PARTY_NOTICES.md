# Third-party notices

ToshLLM is GPL-3.0 (see [LICENSE](LICENSE)). The components below are redistributed inside
the app under their own licenses.

## Math runtime (`Contents/Resources/tosh-sympy`)

Built by `scripts/build-sympy.sh` from pinned releases, each checked against its SHA-256. The
full license texts are in [`helpers/tosh-sympy/licenses`](helpers/tosh-sympy/licenses) and
[`helpers/tosh-scientific/licenses`](helpers/tosh-scientific/licenses), and are copied into
the app at `Contents/Resources/tosh-sympy/licenses`.

| Component | Version | License | Text |
|---|---|---|---|
| SymPy | 1.14.0 | BSD-3-Clause | `LICENSE.sympy.txt` |
| mpmath | 1.3.0 | BSD-3-Clause | `LICENSE.mpmath.txt` |
| NumPy | 2.5.3 | BSD-3-Clause | `LICENSE.numpy.txt` |
| SciPy | 1.18.1 | BSD-3-Clause | `LICENSE.scipy.txt` |
| CPython | 3.13.16 | PSF-2.0 (Python-2.0) | `LICENSE.cpython.txt` |

CPython comes from the [python-build-standalone](https://github.com/astral-sh/python-build-standalone)
release `20261001`, a single executable with these libraries linked in:

| Library | License | Text |
|---|---|---|
| OpenSSL 3 | Apache-2.0 | `LICENSE.openssl-3.txt` |
| SQLite | Public domain | `LICENSE.sqlite.txt` |
| zlib | Zlib | `LICENSE.zlib.txt` |
| bzip2 | bzip2-1.0.6 | `LICENSE.bzip2.txt` |
| liblzma (XZ Utils) | 0BSD | `LICENSE.liblzma.txt` |
| libffi | MIT | `LICENSE.libffi.txt` |
| mpdecimal | BSD-2-Clause | `LICENSE.mpdecimal.txt` |
| Expat | MIT | `LICENSE.expat.txt` |
| libedit | BSD-3-Clause | `LICENSE.libedit.txt` |
| ncurses | X11 | `LICENSE.ncurses.txt` |
| libuuid | BSD-3-Clause | `LICENSE.libuuid.txt` |

The Tcl/Tk libraries and the `_tkinter` and `_dbm` modules of that release are removed at build
time and are not redistributed.

NumPy and SciPy are the official `macosx_14_0` wheels from PyPI. Their BLAS and LAPACK are the
Accelerate framework of macOS, which is part of the system and is not redistributed. The
wheels carry code from other projects under their own terms, all listed in the two license
files above and, for NumPy, in `numpy-components/`. The SciPy wheel also brings these
libraries of the GCC runtime, in `scipy/.dylibs`:

| Library | License | Text |
|---|---|---|
| libgfortran 5 | GPL-3.0-or-later WITH GCC-exception-3.1 | `LICENSE.scipy.txt` |
| libgcc_s 1.1 | GPL-3.0-or-later WITH GCC-exception-3.1 | `LICENSE.scipy.txt` |
| libquadmath 0 | LGPL-2.1-or-later | `LICENSE.lgpl-2.1.txt` |

Their source is GCC, at <https://gcc.gnu.org>. The tests, benchmarks, headers and static
libraries of both wheels are removed at build time and are not redistributed.
