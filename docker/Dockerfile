# AIRR-ML 2025 - Winning Solution Docker Container
# Private Leaderboard Score: 0.69353

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire winning approach
COPY winningApproach/ /app/winningApproach/

# Copy the submission module that adapts to organizers' interface
COPY submission/ /app/submission/

# Set Python path
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Default command - can be overridden
ENTRYPOINT ["python3", "-m", "submission.main"]
