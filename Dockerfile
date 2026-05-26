FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt /app/

# Install python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and other directories
COPY . /app/

# Expose ports for Streamlit (8501) and MLflow (5000) and Jupyter (8888)
EXPOSE 8501 5000 8888

# Set default env variables
ENV PYTHONUNBUFFERED=1

# Default command to run the simulation dashboard
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
