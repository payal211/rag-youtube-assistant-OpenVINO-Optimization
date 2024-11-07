FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTRACE=1 \
    APP_HOME=/app \
    HOME=/app \
    PYTHONPATH=/app \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_THEME_PRIMARY_COLOR="#FF4B4B" \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    SQLITE_DATABASE_PATH=/app/data/sqlite.db \
    LOG_DIR=/app/logs

# Set the working directory
WORKDIR $APP_HOME

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    wget \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and group first
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -ms /bin/bash appuser

# Create all necessary directories with proper permissions
RUN mkdir -p /app/data \
            /app/pages \
            /app/config \
            /app/grafana \
            /app/logs \
            /app/models \
            /app/.streamlit && \
    chown -R appuser:appgroup /app && \
    chmod -R 775 /app && \
    chmod g+s /app/data /app/logs

# Pre-create required files with proper permissions
RUN touch /app/logs/app.log && \
    touch /app/data/sqlite.db && \
    chown appuser:appgroup /app/logs/app.log && \
    chown appuser:appgroup /app/data/sqlite.db && \
    chmod 664 /app/logs/app.log && \
    chmod 664 /app/data/sqlite.db

# Copy requirements and install dependencies
COPY --chown=appuser:appgroup requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=appuser:appgroup app/ ./app/
COPY --chown=appuser:appgroup config/ ./config/
COPY --chown=appuser:appgroup .streamlit/ ./.streamlit/
COPY --chown=appuser:appgroup export_to_onnx.py ./
COPY --chown=appuser:appgroup test_onnx_model.py ./
COPY --chown=appuser:appgroup test_pt_model.py ./
COPY --chown=appuser:appgroup test_ov_model.py ./

# Create healthcheck script
RUN echo '#!/bin/bash\ncurl -f http://localhost:8501/_stcore/health' > /healthcheck.sh && \
    chmod +x /healthcheck.sh && \
    chown appuser:appgroup /healthcheck.sh

# Final permission check
RUN find /app -type d -exec chmod 775 {} + && \
    find /app -type f -exec chmod 664 {} + && \
    chmod +x /healthcheck.sh

# Expose port
EXPOSE 8501

# Switch to non-root user
USER appuser

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["/healthcheck.sh"]

# Run Streamlit
CMD ["streamlit", "run", "app/home.py", "--server.port=8501", "--server.address=0.0.0.0"]
