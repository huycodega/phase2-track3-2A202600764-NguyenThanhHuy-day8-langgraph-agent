FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY data ./data
RUN pip install --no-cache-dir -e '.[dev]'
CMD ["python", "-m", "langgraph_agent_lab.cli", "run-scenarios", "--config", "configs/lab.yaml", "--output", "outputs/metrics.json"]
