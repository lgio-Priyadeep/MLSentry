#!/usr/bin/env bash
set -e

echo "=== Starting MLSentry Ingestion & Monitoring Engine ==="

# Wait for PostgreSQL Database readiness
echo "Waiting for database readiness at ${DATABASE_URL:-postgresql://db:5432/mlsentry}..."
python -c "
import time, os, sys
from urllib.parse import urlparse
import psycopg2

db_url = os.environ.get('DATABASE_URL', 'postgresql://mlsentry:mlsentry_password@db:5432/mlsentry')
parsed = urlparse(db_url)
host = parsed.hostname or 'db'
port = parsed.port or 5432
user = parsed.username or 'mlsentry'
password = parsed.password or 'mlsentry_password'
dbname = parsed.path.lstrip('/') or 'mlsentry'

for attempt in range(1, 31):
    try:
        conn = psycopg2.connect(
            host=host, port=port, user=user, password=password, dbname=dbname, connect_timeout=3
        )
        conn.close()
        print(f'Database is ready (attempt {attempt}).')
        sys.exit(0)
    except Exception as exc:
        print(f'Database not ready yet (attempt {attempt}/30): {exc}')
        time.sleep(2)

print('Database connection timed out after 60 seconds.')
sys.exit(1)
"

# Execute database migrations
echo "Executing database migrations (alembic upgrade head)..."
alembic upgrade head
echo "Database migrations applied successfully."

# Start ASGI application server
echo "Starting Uvicorn ASGI server on port 8000..."
exec uvicorn mlsentry.api.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
