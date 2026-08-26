# ============================================================================
# IreneRewrite — Multi-stage Docker build for CI testing
# ============================================================================
# Usage:
#   docker compose up --build py311-clarabel  # test one job
#   docker compose up --build                 # test all 6 jobs (2 Python x 3 solvers)
#
# Architecture:
#   Stage 1 (base): system-level deps (gmp, mpfr, blas/lapack)
#   Stage 2 (final): per-Python-version image with IreneRewrite + solver backends
#
# Python versions: 3.11 (primary), 3.12 (forward-compat). 3.10 is excluded
# because the pinned scientific stack (numpy/scipy/cvxpy) requires >=3.11.
# ============================================================================

ARG PYTHON_VERSION=3.11

# ---------------------------------------------------------------------------
# Stage 1: Base image with system dependencies
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS base

LABEL org.opencontainers.image.title="IreneRewrite" \
      org.opencontainers.image.description="Polynomial optimization via SOS/SONC/SDP hierarchies" \
      org.opencontainers.image.authors="Mehdi Ghasemi" \
      org.opencontainers.image.version="1.2.5"

# System deps for SymEngine (gmp, mpfr), CVXOPT (blas/lapack), and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgmp10 \
    libmpfr6 \
    libopenblas0-pthread \
    liblapack3 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Stage 2: Final image — Python env + IreneRewrite + test deps
# ---------------------------------------------------------------------------
FROM base AS final

WORKDIR /app

# Pin core scientific stack versions (from Irene/.venv/ baseline).
# NOTE: numpy and scipy are deliberately NOT hard-pinned here — the venv pins
# (numpy==2.4.3, scipy==1.17.1) require Python >=3.11. We float them to the
# best version compatible with the target PYTHON_VERSION so the 3.11/3.12
# matrix builds. (cvxpy==1.9.2 also requires >=3.11, which is why 3.10 is
# excluded from the matrix entirely — see docker-compose.yml.) Every other
# package keeps its exact pin from requirements.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir $(grep -vE '^\s*(#|$|numpy==|scipy==)' requirements.txt) && \
    pip install --no-cache-dir "numpy>=2.0,<3" "scipy>=1.11,<2" && \
    pip install --no-cache-dir pytest pytest-timeout pytest-cov pyyaml

# Copy source tree
COPY Irene/ ./Irene/
COPY tests/ ./tests/
COPY benchmarks/ ./benchmarks/
COPY conftest.py setup.py ./

# Create results directory (mounted as volume in compose)
RUN mkdir -p /app/benchmarks/results

# Healthcheck: verify all solver backends are importable and SymEngine links
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import symengine; import cvxpy; import clarabel; import scs; import cvxopt; print('all solvers OK')" || exit 1

# Entrypoint: run pytest with coverage on the full test matrix
ENTRYPOINT ["python", "-m", "pytest"]
CMD [ \
    "Irene/tests/", \
    "tests/", \
    "-v", \
    "--tb=short", \
    "--timeout=120", \
    "--cov=Irene", \
    "--cov-report=term-missing", \
    "--cov-report=xml:coverage.xml" \
]
