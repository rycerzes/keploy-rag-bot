FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app/

# install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/ 

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# compile bytecode
ENV UV_COMPILE_BYTECODE=1

# uv cache
ENV UV_LINK_MODE=copy

# install deps
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

ENV PYTHONPATH=/app

COPY ./pyproject.toml ./uv.lock /app/

COPY ./*.py /app/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

CMD ["fastapi", "run", "main.py", "--port", "8000", "--host", "0.0.0.0"]