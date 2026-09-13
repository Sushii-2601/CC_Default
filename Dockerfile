FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir --no-deps -e .

# Train the model at build time if no pre-trained artifact was committed.
# Prefers a committed real-data artifact; the 1.6 GB real CSV itself is
# never part of the build context, so the fallback trains on synthetic data.
RUN python -c "from pathlib import Path; import sys; \
    sys.exit(0) if (Path('artifacts/deployment_artifacts_real.joblib').exists() \
    or Path('artifacts/deployment_artifacts.joblib').exists()) else sys.exit(1)" \
    || python scripts/train.py

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py"]
