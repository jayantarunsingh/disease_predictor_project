# Use lightweight Python image
FROM python:3.13-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose Flask's default port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
