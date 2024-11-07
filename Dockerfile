# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_HOME=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR $APP_HOME

# Create a non-root user and group
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -ms /bin/bash appuser

# Create necessary directories with correct permissions
RUN mkdir -p $APP_HOME/data $APP_HOME/logs && \
    chown -R appuser:appgroup $APP_HOME && \
    chmod -R 775 $APP_HOME

# Set the user to appuser (this is key to solving the permission issue)
USER appuser

# Copy the requirements file into the container
COPY requirements.txt ./

# Install Python dependencies
RUN pip install -r requirements.txt

# Create necessary directories
RUN mkdir -p $APP_HOME/pages config data grafana logs $APP_HOME/.streamlit models

# Set permissions for data, logs, and .streamlit directories (specifically for appuser)
RUN chown -R appuser:appgroup $APP_HOME/data $APP_HOME/logs $APP_HOME/.streamlit && \
    chmod -R 775 $APP_HOME/data $APP_HOME/logs $APP_HOME/.streamlit

# Set Python path and Streamlit configs
ENV PYTHONPATH=$APP_HOME/ \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_THEME_PRIMARY_COLOR="#FF4B4B" \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Create empty __init__.py files (ensure directories exist first)
RUN mkdir -p app/pages && \
    touch app/__init__.py app/pages/__init__.py

# Download OpenVINO quantized model from HuggingFace
RUN git clone https://huggingface.co/OpenVINO/Phi-3-mini-128k-instruct-int8-ov

# Copy the application code and other files
COPY app/ ./app/
COPY config/ ./config/
COPY data/ ./data/
COPY grafana/ ./grafana/
COPY .streamlit/config.toml $APP_HOME/.streamlit/config.toml
COPY export_to_onnx.py ./ 
COPY test_onnx_model.py ./ 
COPY test_pt_model.py ./ 
COPY test_ov_model.py ./ 

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Create healthcheck script
RUN echo '#!/bin/bash\ncurl -f http://localhost:8501/_stcore/health' > /healthcheck.sh && \
    chmod +x /healthcheck.sh

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["/healthcheck.sh"]

# Run Streamlit
CMD ["streamlit", "run", "app/home.py", "--server.port=8501", "--server.address=0.0.0.0"]
