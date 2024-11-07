# Use Python 3.9 Slim as base image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Install system dependencies needed for building and managing models
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security and ownership management
RUN useradd -m myuser && chown -R myuser:myuser /app

# Switch to the newly created non-root user
USER myuser

# Add /home/myuser/.local/bin to PATH to fix script location warning
ENV PATH="/home/myuser/.local/bin:${PATH}"

# Copy the requirements.txt file into the container
COPY requirements.txt ./

# Install Python dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories and set permissions for Streamlit and model usage
RUN mkdir -p /app/data /app/pages /app/config /app/grafana /app/logs /app/models /root/.streamlit && \
    chmod -R 777 /app/data /app/pages /app/config /app/grafana /app/logs /app/models

# Set environment variables for Python and Streamlit
ENV PYTHONPATH=/app \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_THEME_PRIMARY_COLOR="#FF4B4B" \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Uncomment the following lines to add OpenVINO and optimum dependencies to requirements
# RUN echo "optimum[openvino]" >> requirements.txt && \
#     echo "transformers" >> requirements.txt && \
#     echo "torch" >> requirements.txt && \
#     echo "openvino" >> requirements.txt

# Download the OpenVINO quantized model from Hugging Face
RUN git clone https://huggingface.co/OpenVINO/Phi-3-mini-128k-instruct-int8-ov /app/models/Phi-3-mini-128k-instruct-int8-ov


# Uncomment the following lines if you need to download and process the model:
# WORKDIR /app/models
# RUN git lfs install && \
#     git clone https://huggingface.co/microsoft/Phi-3-mini-128k-instruct && \
#     optimum-cli export openvino \
#     --model "Phi-3-mini-128k-instruct" \
#     --task text-generation-with-past \
#     --weight-format int4 \
#     --group-size 128 \
#     --ratio 0.6 \
#     --sym \
#     --trust-remote-code /app/models/Phi-3-mini-128k-instruct-int4-ov


# Set the working directory to /app (for later copy)
WORKDIR /app

# Copy the application code, configurations, and other files to the container
COPY app/ ./app/
COPY config/ ./config/
COPY data/ ./data/
COPY grafana/ ./grafana/
COPY .streamlit/config.toml /root/.streamlit/config.toml
COPY export_to_onnx.py ./
COPY test_onnx_model.py ./
COPY test_pt_model.py ./
COPY test_ov_model.py ./

# Expose Streamlit's default port (8501)
EXPOSE 8501

# Health check script
RUN echo '#!/bin/bash\ncurl -f http://localhost:8501/_stcore/health' > /app/healthcheck.sh && \
    chmod +x /app/healthcheck.sh

# Define health check for the container
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["/app/healthcheck.sh"]

# Default command to run Streamlit app
CMD ["streamlit", "run", "app/home.py", "--server.port=8501", "--server.address=0.0.0.0"]
