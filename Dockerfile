# Container image for the FastAPI server example. [OPTIONAL FEATURE:
# delete this file + .dockerignore + compose.yaml + the README "Docker"
# section to remove]
#
# Runs examples/fastapi_server.py, which is the deployable surface: the
# CLI in main.py is for local development. To containerise a different
# entry point, change the CMD.
#
# Build and run:
#   docker build -t my-agent .
#   docker run --rm -p 8000:8000 --env-file .env my-agent
#
# Secrets come from the environment (--env-file / your orchestrator's
# secret store) — never COPY a .env into the image; .dockerignore
# excludes it so an accidental `COPY . .` cannot bake one in.

FROM python:3.12-slim

# Fail fast, no .pyc clutter, unbuffered logs so `docker logs` is live.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependency layer first: it only re-installs when packaging metadata
# changes, so source edits rebuild in seconds.
COPY pyproject.toml README.md ./
COPY app app
COPY lib lib
COPY tools tools
RUN pip install --no-cache-dir ".[serve]"

# The server example and the CLI module it imports STREAMING_NODES from.
COPY examples examples
COPY main.py ./

# Run as a non-root user: a container escape should not land as root.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# JSON logs on stderr — Docker/Kubernetes/Cloud Logging collectors parse
# them natively (see lib/log.py).
ENV LOG_FORMAT=json

CMD ["python", "-m", "examples.fastapi_server"]
