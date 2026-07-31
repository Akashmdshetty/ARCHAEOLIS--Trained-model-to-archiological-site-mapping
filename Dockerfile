# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install ONLY Streamlit so the container can start quickly
# This skips heavy ML libraries since the goal is just to show a working Dockerfile
RUN pip install streamlit

# Copy the rest of the application code into the container
COPY . .

# Expose the port Streamlit uses by default
EXPOSE 8501

# Command to run the simple Streamlit dashboard
# Uses --server.headless=true to avoid trying to open a browser inside Docker
CMD ["streamlit", "run", "dashboard/simple_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
