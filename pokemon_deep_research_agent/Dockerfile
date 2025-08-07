# Multi-platform Dockerfile for Pokemon Deep Research Agent
# Supports both AMD64 and ARM64 (MacBook M3 Pro)
FROM --platform=$BUILDPLATFORM python:3.11-slim-bullseye

# Set build arguments for multi-platform support
ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Create non-root user for security
RUN groupadd -r convergence && useradd -r -g convergence convergence_user

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH="/app/src"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements/ ./requirements/
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY main.py ./
COPY demo.py ./
COPY run_tests.py ./
COPY config.json ./

# Create necessary directories
RUN mkdir -p data/cache logs && \
    chown -R convergence_user:convergence /app

# Switch to non-root user
USER convergence_user

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "main.py", "--interactive"]

