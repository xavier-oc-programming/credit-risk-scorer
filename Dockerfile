FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# For Azure App Service deployment, gunicorn with uvicorn workers
# is used instead of uvicorn directly — see startup.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
