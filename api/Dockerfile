FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir "mlflow<3"

ARG MLFLOW_TRACKING_URI
ARG MLFLOW_TRACKING_USERNAME
COPY download_model.py .
RUN --mount=type=secret,id=dagshub_token,env=MLFLOW_TRACKING_PASSWORD,required=true \
    python download_model.py

RUN pip install --no-cache-dir -r model/requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]