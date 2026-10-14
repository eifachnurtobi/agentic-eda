# Sandbox image for generated code. Built through compose.yaml (`docker compose build`).
# The base image is pinned by version; each run's run.json records the ID of the image
# it actually used.
FROM python:3.12.7-slim

COPY requirements.lock /opt/eda/requirements.lock
# The freeze keeps every installed version, transitive ones included, inspectable in the image.
RUN pip install --no-cache-dir -r /opt/eda/requirements.lock \
    && pip freeze > /opt/eda/requirements.freeze.txt

# The trusted notebook executor (nb_executor.py) is mounted read-only at /opt/eda/ for each
# notebook execution, so changing it needs no rebuild.

WORKDIR /work
