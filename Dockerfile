FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (including local emergentintegrations package)
COPY . .

# Expose port (Render default)
EXPOSE 10000

# Start command
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "10000"]
