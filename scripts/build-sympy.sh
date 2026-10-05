#!/bin/zsh
# Builds the self-contained math runtime in vendor/tosh-sympy: a pinned CPython, SymPy and
# mpmath for the symbolic tools, NumPy and SciPy for the scientific ones, and the helpers
# from helpers/. Nothing here uses a Python from the system, and every download is checked
# against a pinned SHA-256.
#
#   ./scripts/build-sympy.sh              # host architecture
#   ARCH=x86_64 ./scripts/build-sympy.sh  # cross-build (CI on Apple Silicon runners)
set -e
cd "$(dirname "$0")/.."
ROOT="$PWD"
ARCH="${ARCH:-$(uname -m)}"

PYTHON_VERSION="3.13.16"
PYTHON_RELEASE="20261001"   # astral-sh/python-build-standalone
SYMPY_VERSION="1.14.0"
MPMATH_VERSION="1.3.0"
NUMPY_VERSION="2.5.3"
SCIPY_VERSION="1.18.1"

typeset -A PYTHON_SHA256=(
    x86_64 a88bef59d9dd61ba4210772cce57a4b4cb963aa745ab956f7cc79ba60f9e2523
    arm64  d00669acb53c1b014f1fcf5eaea740d8e45c3aff4b1e0244dc0f8697fb211a82
)
SYMPY_SHA256="e091cc3e99d2141a0ba2847328f5479b05d94a6635cb96148ccb3f34671bd8f5"
MPMATH_SHA256="a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c"
# The macOS 14 wheels: BLAS and LAPACK come from Accelerate, which is part of the system,
# instead of a bundled OpenBLAS.
typeset -A NUMPY_SHA256=(
    x86_64 ccbc4665079665c3cf3bab4db9f6b095370cd6437d66be549b6c2a1fd19e1958
    arm64  f9a2353b37a1a9e78fd82b27ad7e2a32a2d036604d18f02b05e3136c62ca3b09
)
typeset -A SCIPY_SHA256=(
    x86_64 75b00eb8fb802090aa903f4ea1c7f5a584779f967361e68b7e98e531cc2d7174
    arm64  ea324d9dd34c38bfb9bec8ca4d1b407db97dbb74029f566b8e322b1b6fe56fe6
)

OUT="$ROOT/vendor/tosh-sympy"
CACHE="$ROOT/vendor/.downloads"
STAMP="$PYTHON_VERSION+$PYTHON_RELEASE sympy-$SYMPY_VERSION mpmath-$MPMATH_VERSION numpy-$NUMPY_VERSION scipy-$SCIPY_VERSION $ARCH"
mkdir -p "$CACHE"

fetch() {
    local url="$1" sha="$2" file="$CACHE/${1:t}"
    if [ ! -f "$file" ] || [ "$(shasum -a 256 "$file" | cut -d' ' -f1)" != "$sha" ]; then
        curl -fL --retry 4 --retry-delay 5 -o "$file" "$url"
    fi
    if [ "$(shasum -a 256 "$file" | cut -d' ' -f1)" != "$sha" ]; then
        echo "ERROR: checksum mismatch for ${file:t}" >&2
        rm -f "$file"
        exit 1
    fi
    echo "$file"
}

python_tarball() {
    local triple="$1"
    [ "$triple" = "arm64" ] && triple="aarch64"
    fetch "https://github.com/astral-sh/python-build-standalone/releases/download/$PYTHON_RELEASE/cpython-$PYTHON_VERSION+$PYTHON_RELEASE-$triple-apple-darwin-install_only_stripped.tar.gz" \
          "${PYTHON_SHA256[$1]}"
}

native_wheel() {
    local name="$1" version="$2" arch="$3" sha="$4"
    fetch "https://files.pythonhosted.org/packages/cp313/${name[1]}/$name/$name-$version-cp313-cp313-macosx_14_0_$arch.whl" "$sha"
}

# A wheel may keep search paths of the machine it was built on; only relative ones stay.
relative_rpaths() {
    local binary rpath changed
    for binary in "$1"/(numpy|scipy)/**/*.(so|dylib)(.ND); do
        changed=0
        for rpath in $(otool -l "$binary" | awk '/cmd LC_RPATH/{f=1} f&&/^ *path /{print $2; f=0}' | sort -u); do
            case "$rpath" in
                @*) ;;
                *) while install_name_tool -delete_rpath "$rpath" "$binary" 2>/dev/null; do changed=1; done ;;
            esac
        done
        [ "$changed" = 0 ] || codesign --force -s - "$binary"
    done
}

case "$ARCH" in
    x86_64|arm64) BASE_ARCH="$ARCH" ;;
    universal)    BASE_ARCH="x86_64" ;;
    *) echo "ERROR: unsupported ARCH '$ARCH'" >&2; exit 1 ;;
esac

SYMPY_WHEEL="$(fetch "https://files.pythonhosted.org/packages/py3/s/sympy/sympy-$SYMPY_VERSION-py3-none-any.whl" "$SYMPY_SHA256")"
MPMATH_WHEEL="$(fetch "https://files.pythonhosted.org/packages/py3/m/mpmath/mpmath-$MPMATH_VERSION-py3-none-any.whl" "$MPMATH_SHA256")"
PYTHON_TARBALL="$(python_tarball "$BASE_ARCH")"

rm -rf "$OUT"
mkdir -p "$OUT"
tar xzf "$PYTHON_TARBALL" -C "$OUT"

PY="$OUT/python"
LIB="$PY/lib/python${PYTHON_VERSION%.*}"

if [ "$ARCH" = "universal" ]; then
    SLICE="$(mktemp -d)"
    tar xzf "$(python_tarball arm64)" -C "$SLICE" "python/bin/python${PYTHON_VERSION%.*}"
    lipo -create "$PY/bin/python${PYTHON_VERSION%.*}" "$SLICE/python/bin/python${PYTHON_VERSION%.*}" \
         -output "$PY/bin/python.universal"
    mv "$PY/bin/python.universal" "$PY/bin/python${PYTHON_VERSION%.*}"
    rm -rf "$SLICE"
fi

# The interpreter is one static executable. Everything below is not needed by the helpers.
# unittest, pydoc and _pyrepl stay because NumPy and SciPy import them.
find "$PY/bin" -mindepth 1 ! -name "python${PYTHON_VERSION%.*}" -delete
mv "$PY/bin/python${PYTHON_VERSION%.*}" "$PY/bin/python3"
rm -rf "$PY/include" "$PY/share" "$PY/lib/pkgconfig" "$PY/lib"/lib*.dylib \
       "$PY/lib"/(itcl|tcl|tk|thread)*(N) \
       "$LIB/lib-dynload"/* "$LIB/site-packages"/* "$LIB"/config-* \
       "$LIB"/(test|idlelib|tkinter|turtledemo|ensurepip|venv|pydoc_data|lib2to3|curses|dbm|sqlite3|wsgiref|xmlrpc)(N) \
       "$LIB"/(turtle|doctest|pdb|smtplib|ftplib|imaplib|poplib|mailbox|webbrowser|tarfile|zipapp).py(N)
find "$LIB" -name '__pycache__' -type d -prune -exec rm -rf {} +

unzip -q "$SYMPY_WHEEL" -d "$LIB/site-packages"
unzip -q "$MPMATH_WHEEL" -d "$LIB/site-packages"
unzip -q "$(native_wheel numpy "$NUMPY_VERSION" "$BASE_ARCH" "${NUMPY_SHA256[$BASE_ARCH]}")" -d "$LIB/site-packages"
unzip -q "$(native_wheel scipy "$SCIPY_VERSION" "$BASE_ARCH" "${SCIPY_SHA256[$BASE_ARCH]}")" -d "$LIB/site-packages"
relative_rpaths "$LIB/site-packages"

if [ "$ARCH" = "universal" ]; then
    # every extension module and bundled library gets the arm64 slice of the same file
    SLICE="$(mktemp -d)"
    unzip -q "$(native_wheel numpy "$NUMPY_VERSION" arm64 "${NUMPY_SHA256[arm64]}")" -d "$SLICE"
    unzip -q "$(native_wheel scipy "$SCIPY_VERSION" arm64 "${SCIPY_SHA256[arm64]}")" -d "$SLICE"
    relative_rpaths "$SLICE"
    for binary in "$LIB/site-packages"/(numpy|scipy)/**/*.(so|dylib)(.ND); do
        other="$SLICE/${binary#$LIB/site-packages/}"
        if [ ! -f "$other" ]; then
            echo "ERROR: ${binary#$LIB/site-packages/} has no arm64 counterpart" >&2
            exit 1
        fi
        lipo -create "$binary" "$other" -output "$binary.universal"
        mv "$binary.universal" "$binary"
    done
    rm -rf "$SLICE"
fi

# tests, headers, stubs and static libraries are for developing against the packages
find "$LIB/site-packages" -type d \( -name tests -o -name benchmarks -o -name include \) -prune -exec rm -rf {} +
find "$LIB/site-packages" -type f \( -name '*.pyi' -o -name '*.pxd' -o -name '*.pyx' -o -name '*.h' -o -name '*.a' \
     -o -name '*.c' -o -name '*.typed' \) -delete
rm -rf "$LIB/site-packages"/*.dist-info "$LIB/site-packages"/isympy.py "$LIB/site-packages"/bin(N)

mkdir -p "$OUT/tosh_sympy" "$OUT/tosh_scientific"
cp "$ROOT"/helpers/tosh-sympy/tosh_sympy/*.py "$OUT/tosh_sympy/"
cp "$ROOT"/helpers/tosh-scientific/tosh_scientific/*.py "$OUT/tosh_scientific/"
cp -R "$ROOT/helpers/tosh-sympy/licenses" "$OUT/licenses"
cp -R "$ROOT"/helpers/tosh-scientific/licenses/* "$OUT/licenses/"

# The app bundle is read-only and signed, so the bytecode is built here and never checked
# against the sources at run time. compileall needs its own sources, so it runs before they go.
BUILD_PY="$PY/bin/python3"
if ! "$BUILD_PY" -I -c 'pass' 2>/dev/null; then
    # cross-build: the target interpreter cannot run here, so use the host one of the same release
    HOST="$(mktemp -d)"
    tar xzf "$(python_tarball "$(uname -m)")" -C "$HOST"
    BUILD_PY="$HOST/python/bin/python${PYTHON_VERSION%.*}"
fi
"$BUILD_PY" -I -m compileall -q -b -j 0 --invalidation-mode unchecked-hash "$LIB"
# the helpers keep their sources, so their bytecode is checked against them: a copied edit is never shadowed
"$BUILD_PY" -I -m compileall -q -j 0 --invalidation-mode checked-hash "$OUT/tosh_sympy" "$OUT/tosh_scientific"
[ -n "$HOST" ] && rm -rf "$HOST"
# the libraries ship as bytecode only: half the size and half the files to seal in the signature
find "$LIB" -name '*.py' -delete

echo "$STAMP" > "$OUT/VERSION"
echo "math runtime ready at $OUT ($(du -sh "$OUT" | cut -f1), $STAMP)"
