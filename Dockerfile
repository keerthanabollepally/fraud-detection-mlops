# Use python base image
FROM python:3.11-slim

# Create working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose API port
EXPOSE 8000

# Command to run API
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
