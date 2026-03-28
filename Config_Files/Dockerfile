# Use the official Python 3.10 slim image as the base for a smaller footprint
FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc files
# and to ensure output is sent directly to the terminal (unbuffered)
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install essential system dependencies required by OpenCV and other libraries
# libgl1-mesa-glx and libglib2.0-0 are critical for CV2 to function on Linux
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    espeak \
    espeak-ng \
    alsa-utils \
    pulseaudio \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker's caching mechanism
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir keeps the final image size small
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container
COPY . .

# Expose the default port used by Streamlit (8501)
EXPOSE 8501

# Configure Streamlit to run in a headless environment for deployment
# Headless mode prevents the browser from trying to open automatically on the server
# enableCORS=false helps prevent "https" and "cross-origin" errors during live testing
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--server.enableCORS=false", \
            "--server.enableXsrfProtection=false"]