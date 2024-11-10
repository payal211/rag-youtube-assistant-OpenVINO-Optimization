services:
  elasticsearch:
    image: elasticsearch-rag:latest
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
      - bootstrap.memory_lock=true
    ports:
      - 9200:9200
      - 9300:9300
    volumes:
      - esdata:/usr/share/elasticsearch/data

  app:
    build: .
    image: youtube-rag:latest
    environment:
      - ELASTICSEARCH_HOST=elasticsearch
      - ELASTICSEARCH_PORT=9200
      - YOUTUBE_API_KEY=${YOUTUBE_API_KEY}
      - PYTHONPATH=/app
      - STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
      - STREAMLIT_THEME_PRIMARY_COLOR="#FF4B4B"
      - SQLITE_DATABASE_PATH=/app/data/sqlite.db
      - LOG_DIR=/app/logs
    ports:
      - 8501:8501
    volumes:
      - app-data:/app/data
      - app-logs:/app/logs
      - app-config:/app/config
    depends_on:
      - elasticsearch

  grafana:
    image: grafana-rag:latest
    ports:
      - 3000:3000
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=false
      - GF_SECURITY_ADMIN_USER=${GRAFANA_USERNAME:-admin}
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
      - GF_INSTALL_PLUGINS=frser-sqlite-datasource
      - GF_PLUGINS_ALLOW_LOADING_UNSIGNED_PLUGINS=frser-sqlite-datasource
    volumes:
      - grafana-storage:/var/lib/grafana
      - app-data:/app/data:ro
    depends_on:
      - elasticsearch
      - app

volumes:
  esdata:
  app-data:
  app-logs:
  app-config:
  grafana-storage:
