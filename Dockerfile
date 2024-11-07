FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and assign ownership to the /app directory
RUN useradd -m myuser && chown -R myuser:myuser /app

# Switch to the new user
USER myuser

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Create necessary directories with the appropriate permissions
RUN mkdir -p /app/data /app/pages /app/config /app/grafana /app/logs /app/models /app/.streamlit && \
    chmod -R 777 /app/data /app/pages /app/config /app/grafana /app/logs /app/models /app/.streamlit

# Set environment variables
ENV PYTHONPATH=/app \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_THEME_PRIMARY_COLOR="#FF4B4B" \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Download the OpenVINO quantized model
RUN git clone https://huggingface.co/OpenVINO/Phi-3-mini-128k-instruct-int8-ov

# Set the working directory to /app
WORKDIR /app

# Copy the application code and other files
COPY app/ ./app/
COPY config/ ./config/
COPY data/ ./data/
COPY grafana/ ./grafana/
COPY .streamlit/config.toml /root/.streamlit/config.toml
COPY export_to_onnx.py ./
COPY test_onnx_model.py ./
COPY test_pt_model.py ./
COPY test_ov_model.py ./

# Expose port 8501 for Streamlit
EXPOSE 8501

# Health check script
RUN echo '#!/bin/bash\ncurl -f http://localhost:8501/_stcore/health' > /healthcheck.sh && \
    chmod +x /healthcheck.sh

# Healthcheck definition
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["/healthcheck.sh"]

# Run Streamlit
CMD ["streamlit", "run", "app/home.py", "--server.port=8501", "--server.address=0.0.0.0"]
