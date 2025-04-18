# Use an official Python runtime as a parent image
# Using a slim variant to reduce image size
FROM python:3.11-slim

# Set environment variables
# Prevents Python from writing pyc files to disc (equivalent to python -B)
ENV PYTHONDONTWRITEBYTECODE 1
# Prevents Python from buffering stdout and stderr (equivalent to python -u)
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by Python packages
# (Example: build-essential for C extensions). Add others if build fails.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir reduces image size slightly
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
# Copy backend first to potentially improve layer caching if only frontend changes
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Note: We don't need EXPOSE or CMD here, as heroku.yml will define the run commands. 