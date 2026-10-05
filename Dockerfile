# syntax=docker/dockerfile:1

# Build Python dependencies separately so pip tooling does not become part of
# the production image.
FROM python:3.12-slim-bookworm AS builder

# Copy the dependency manifest first to preserve this cached layer when only
# application source files change.
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir \
    --target /opt/python \
    --requirement requirements.txt


# Use the same small official image for a predictable Python runtime.
FROM python:3.12-slim-bookworm AS runtime

# LightGBM requires the OpenMP runtime. Avoid recommended packages and remove
# apt metadata in the same layer to keep the final image small.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /app --no-create-home app

ENV PYTHONPATH=/opt/python \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Bring in only installed dependencies and the files needed by the API at
# runtime; tests, documentation, secrets, and local tooling stay outside.
COPY --from=builder /opt/python /opt/python
COPY --chown=app:app api ./api
COPY --chown=app:app db ./db
COPY --chown=app:app models ./models
COPY --chown=app:app web ./web

# Run the service without root privileges.
USER app

EXPOSE 8000

# Bind to every container interface; configure the database at runtime with
# V4W_DATABASE_URL rather than baking credentials into the image.
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
