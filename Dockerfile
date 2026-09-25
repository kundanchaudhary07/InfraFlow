# =============================================================================
# DeployGuard Demo — Order Service
# Phase 2: Dockerization
#
# Build:  docker build -t deployguard-demo:phase2 .
# Run:    docker run --rm -p 8000:8000 deployguard-demo:phase2
# =============================================================================

# -----------------------------------------------------------------------------
# Base image
# -----------------------------------------------------------------------------
# python:3.13-slim is the official slim variant: full CPython runtime,
# minus documentation, test suites, and most optional system libraries.
# It gives us a minimal, well-maintained attack surface without resorting
# to Alpine (which uses musl libc and can cause subtle compatibility issues
# with compiled Python extensions like pydantic-core).
FROM python:3.13-slim

# -----------------------------------------------------------------------------
# Runtime environment variables
# -----------------------------------------------------------------------------
# PYTHONDONTWRITEBYTECODE  — prevents Python from writing .pyc files to disk.
#                            Keeps the image clean and avoids stale bytecode.
# PYTHONUNBUFFERED         — forces stdout/stderr to be unbuffered, so logs
#                            appear in real time in `docker logs` output.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# -----------------------------------------------------------------------------
# Working directory
# -----------------------------------------------------------------------------
# /app is a conventional, unambiguous location for application code.
WORKDIR /app

# -----------------------------------------------------------------------------
# Non-root user
# -----------------------------------------------------------------------------
# Running as root inside a container is a significant security risk:
# any container escape bug grants full host access.
# We create a dedicated system user (no home dir, no login shell) and
# switch to it before starting the application.
RUN adduser \
        --system \
        --no-create-home \
        --group \
        appuser

# -----------------------------------------------------------------------------
# Dependency installation (cache-optimised layer ordering)
# -----------------------------------------------------------------------------
# We copy ONLY the runtime requirements file first. Docker layer caching means
# this layer — and the expensive pip install — is reused on every subsequent
# build as long as requirements-runtime.txt has not changed, even if
# application source has changed. This is the single most important
# Dockerfile caching technique.
#
# We install from requirements-runtime.txt, NOT requirements-dev.txt.
# Test tools (pytest, httpx) are never copied into the production image.
COPY requirements-runtime.txt .

# --no-cache-dir          : pip's HTTP cache is useless inside a build layer;
#                           omitting it reduces image size.
# --root-user-action=ignore: we run pip as root during the build step (before
#                           switching to appuser). This suppresses a spurious
#                           pip warning — it is not a security concern because
#                           we immediately drop privileges at runtime.
RUN pip install --upgrade pip --no-cache-dir --root-user-action=ignore \
    && pip install --no-cache-dir --root-user-action=ignore \
       -r requirements-runtime.txt

# -----------------------------------------------------------------------------
# Application source
# -----------------------------------------------------------------------------
# Copied after pip install so that source-only changes do not invalidate
# the dependency layer above.
COPY app/ ./app/

# -----------------------------------------------------------------------------
# Ownership
# -----------------------------------------------------------------------------
# Transfer ownership of the working directory to the non-root user before
# we switch to it, so the process can read its own files.
RUN chown -R appuser:appuser /app

USER appuser

# -----------------------------------------------------------------------------
# Port declaration
# -----------------------------------------------------------------------------
# Informs Docker (and future orchestrators) that the container listens on
# port 8000. This is documentation — it does not publish the port to the host;
# that is done with -p at `docker run` time.
EXPOSE 8000

# -----------------------------------------------------------------------------
# Runtime command
# -----------------------------------------------------------------------------
# We use the exec form (JSON array) so the process receives OS signals
# directly (SIGTERM for graceful shutdown) rather than via a shell wrapper.
#
# Key flags:
#   --host 0.0.0.0   — bind to all interfaces inside the container so traffic
#                      forwarded from the host port mapping is accepted.
#                      The default 127.0.0.1 would be unreachable from outside.
#   --port 8000      — match the EXPOSE declaration above.
#   --workers 1      — one process is appropriate for a demo/single-container
#                      deployment. Multiple workers require shared-memory
#                      coordination (Phase N concern).
#   No --reload      — hot-reload watches the filesystem and restarts the
#                      process; it is a development convenience, not suitable
#                      for production containers.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
