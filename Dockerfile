# ============================================================================
# IreneRewrite — Multi-stage Docker build for CI testing
# ============================================================================
# Usage:
#   docker-compose up --build python310  # test on Python 3.10 only
#   docker-compose up --build            # test all three versions in parallel
#
# Architecture:
#   Stage 1 (base): system-level deps (gmp, mpfr, blas/lapack)
#   Stage 2 (final): per-Python-version image with IreneRewrite + solver backends
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

# Pin core scientific stack versions (from Irene/.venv/ baseline)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
