# OptimBench test/dev image
FROM python:3.11-slim

# uv: fast installer and venv manager
RUN pip install --no-cache-dir uv

WORKDIR /app
COPY . /app

# In-project virtual env with the dev and media extras. rl (PyTorch) is left out
# on purpose: it is heavy and only needed to train or run the learned agent.
RUN uv venv /app/.venv \
    && uv pip install --python /app/.venv/bin/python -e ".[dev]"

ENV PATH="/app/.venv/bin:$PATH"
# Headless SDL so pygame renders without a display.
ENV SDL_VIDEODRIVER=dummy

CMD ["pytest", "-q"]
