# Use Python 3.12.3 image
FROM python:3.12.3-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose the Flask port (update this if using a different port)
EXPOSE 8000

# Command to run the Flask application
CMD ["python", "app.py"]
