# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: builder
#   Creates a self-contained virtualenv at /venv and installs all production
#   dependencies (including the optional [llm] extra) via pip.  Using a venv
#   avoids the prefix-install + poetry-export approach that silently drops
#   transitive dependencies (packaging, shellingham, etc.) that Poetry treats
#   as pip-internal but mera/typer need as importable runtime modules.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        git \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Upgrade pip inside the venv so it picks up the latest resolver.
RUN pip install --upgrade pip

# Copy only the files needed for dependency resolution first.
# This layer is cached unless pyproject.toml or README.md change.
WORKDIR /build
COPY pyproject.toml README.md ./

# Install all production dependencies including the [llm] extra (transformers).
# pip resolves the full transitive closure from pyproject.toml, so no packages
# are silently omitted.
# We copy a minimal stub src/ so pyproject.toml's package discovery succeeds
# without pulling in the real source yet (keeps the dep layer cacheable).
RUN mkdir -p src/sakura_simulator && \
    touch src/sakura_simulator/__init__.py && \
    pip install --no-compile ".[llm]"

# Now copy the real source and reinstall only the package itself (deps are
# already cached in the layer above).
COPY src/ ./src/
RUN pip install --no-compile --no-deps .

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: runtime
#   Minimal bookworm-slim base plus the system libs required by mera's native
#   extensions (libmeradna-full.x86.so, libmeratvm.so).  No build tools, no
#   dev dependencies, no test code.  The entire Python environment is in /venv.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim-bookworm AS runtime

# Runtime system libraries required by mera's native shared objects.
# These are NOT bundled inside the wheels and must come from apt:
#   libgoogle-glog0v6     → libglog.so.0
#   libgflags2.2          → libgflags.so.2.2
#   libgomp1              → libgomp.so.1  (OpenMP threading in mera-tvm)
#   libunwind8            → libunwind.so.8
#   libllvm14             → libLLVM-14.so.1  (mera TVM links against LLVM 14)
#   libedit2              → libedit.so.2
#   libbsd0               → libbsd.so.0  (transitive dep of libedit2)
#
# libdnnl.so.2 is shipped inside the onednn-cpu-gomp wheel; it lives inside
# the venv at /venv/lib/libdnnl.so.2.  LD_LIBRARY_PATH exposes it to the
# dynamic linker.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgoogle-glog0v6 \
        libgflags2.2 \
        libgomp1 \
        libunwind8 \
        libllvm14 \
        libedit2 \
        libbsd0 \
        ca-certificates \
        gcc \
        libc6-dev \
    # mera-tvm-full requires GLIBCXX_3.4.32 (GCC 13.2), but Debian bookworm ships
    # GCC 12 (max GLIBCXX_3.4.30).  Pull libstdc++6 from Debian trixie — it is a
    # drop-in upgrade (same SONAME libstdc++.so.6, strict backward compatibility).
    && echo "deb https://deb.debian.org/debian trixie main" \
        > /etc/apt/sources.list.d/trixie.list \
    && apt-get update \
    && apt-get install -y -t trixie --no-install-recommends libstdc++6 \
    && rm /etc/apt/sources.list.d/trixie.list \
    # bookworm ships libglog.so.0.6.0 with SONAME=1 (libglog.so.1), but mera-tvm
    # was linked against SONAME=0 (libglog.so.0).  Create the missing alias.
    && ln -sf /usr/lib/x86_64-linux-gnu/libglog.so.0.6.0 \
              /usr/lib/x86_64-linux-gnu/libglog.so.0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the complete virtualenv from the builder stage.
COPY --from=builder /venv /venv

# torch 1.12.1 (pulled in by mera[full]) marks libtorch_cpu.so with an
# executable-stack PT_GNU_STACK entry.  Linux 6.x refuses to load such
# libraries with mprotect(PROT_EXEC) without CONFIG_READ_IMPLIES_EXEC, causing
# an ImportError at runtime.  Clear the PF_X bit in-place via a base64-encoded
# Python one-shot — no extra tools required, and the library works correctly.
RUN echo aW1wb3J0IHN0cnVjdCxvcwpkZWYgYyhwKToKICAgIHRyeToKICAgICAgICB3aXRoIG9wZW4ocCwicitiIikgYXMgZjoKICAgICAgICAgICAgaWYgZi5yZWFkKDQpIT1iIn9FTEYiOnJldHVybgogICAgICAgICAgICBpZiBzdHJ1Y3QudW5wYWNrKCJCIixmLnJlYWQoMSkpWzBdIT0yOnJldHVybgogICAgICAgICAgICBmLnNlZWsoMCk7aD1mLnJlYWQoNjQpCiAgICAgICAgICAgIG89c3RydWN0LnVucGFja19mcm9tKCI8USIsaCwzMilbMF0KICAgICAgICAgICAgZT1zdHJ1Y3QudW5wYWNrX2Zyb20oIjxIIixoLDU0KVswXQogICAgICAgICAgICBuPXN0cnVjdC51bnBhY2tfZnJvbSgiPEgiLGgsNTYpWzBdCiAgICAgICAgICAgIGZvciBpIGluIHJhbmdlKG4pOgogICAgICAgICAgICAgICAgZi5zZWVrKG8raSplKTtwaD1mLnJlYWQoZSkKICAgICAgICAgICAgICAgIGlmIHN0cnVjdC51bnBhY2tfZnJvbSgiPEkiLHBoLDApWzBdPT0weDY0NzRlNTUxOgogICAgICAgICAgICAgICAgICAgIGZsPXN0cnVjdC51bnBhY2tfZnJvbSgiPEkiLHBoLDQpWzBdCiAgICAgICAgICAgICAgICAgICAgaWYgZmwmMTpmLnNlZWsobytpKmUrNCk7Zi53cml0ZShzdHJ1Y3QucGFjaygiPEkiLGZsJn4xKSkKICAgIGV4Y2VwdDpwYXNzCltjKG9zLnBhdGguam9pbihyLGZuKSkgZm9yIHIsXyxmcyBpbiBvcy53YWxrKCIvdmVudiIpIGZvciBmbiBpbiBmcyBpZiAiLnNvIiBpbiBmbl0K | base64 -d | python3

# Put the venv's bin/ on PATH so sakura, streamlit, and python resolve to the
# venv binaries rather than the system Python.
ENV PATH="/venv/bin:$PATH"

# meratvm_internal._cpp_lib_load resolves the mera native library by searching
# for filenames keyed on the detected host arch string.  Inside Docker containers
# query_arch returns "aarch64" (a known mera SDK quirk — platform.processor()
# returns '' so the detection falls back to the wrong default).  The actual
# library IS the x86 build; creating "aarch64"-named aliases in /venv/lib/ (the
# first candidate directory the loader checks) lets the lookup succeed while the
# x86_64 binary runs correctly on the x86_64 host.
RUN ln -sf /venv/lib/python3.10/site-packages/tvm/libmeradna-full.x86.so \
           /venv/lib/libmeradna-full.aarch64.so \
 && ln -sf /venv/lib/python3.10/site-packages/tvm/libmeradna-full.x86.so \
           /venv/lib/libmeradna-host-only.aarch64.so \
 && ln -sf /venv/lib/python3.10/site-packages/tvm/libmeradna-full.x86.so \
           /venv/lib/libmeradna-runtime.aarch64.so

# Make the dynamic linker find libdnnl.so.2 (installed by onednn-cpu-gomp
# into /venv/lib/) and the mera native libs aliased above.
ENV LD_LIBRARY_PATH=/venv/lib

# Prevent TensorFlow (pulled in by mera[full]) from initialising its protobuf
# descriptor pool on import — avoids conflicts with the transformers library.
# tokenizer.py also sets this, but the Dockerfile ENV covers all sub-processes.
ENV USE_TF=0

# Streamlit must bind to 0.0.0.0 inside Docker or the port is unreachable from
# the host.  --server.headless=true suppresses the interactive TTY prompt that
# would otherwise hang the container on startup.
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

# All relative paths in configs/models.yaml (models/, artifacts/, tokenizers/)
# resolve from the working directory.  Bind mounts in docker-compose populate
# these subdirectories at runtime.
WORKDIR /workspace

ENTRYPOINT ["sakura"]
CMD ["--help"]
