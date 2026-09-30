FROM python:3.10-slim

WORKDIR /app

# Install dependencies first to cache the layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Set PYTHONPATH
ENV PYTHONPATH=/app/src

# Expose Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "src/tracelens/app/demo.py", "--server.port=8501", "--server.address=0.0.0.0"]
