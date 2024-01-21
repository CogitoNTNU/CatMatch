# Dockerfile
FROM python:3.11.5
RUN pip install poetry
COPY . .
RUN poetry install
# To deploy to Railway, you have to set the PORT environment variable to 8000
# 
ENTRYPOINT ["poetry", "run", "uvicorn", "catmatch.serve:app", "--host", "0.0.0.0", "--port", "8000"]
