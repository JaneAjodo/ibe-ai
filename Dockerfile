FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY sample_data.xlsx .

# Create ChromaDB persistence directory
RUN mkdir -p /app/chroma_db

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
