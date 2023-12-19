# Base Image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Create a non-root group and user
RUN groupadd -r nonroot && useradd -r -g nonroot -d /app nonroot

# Create temp and .cache directories and change ownership to nonroot user
RUN mkdir temp .cache && chown -R nonroot:nonroot /app

# Set HOME environment variable to /app so that cache is stored in /app/.cache
ENV HOME=/app

# Python outputs everything directly to the terminal (e.g. docker logs) so that it can be observed in real-time
ENV PYTHONUNBUFFERED=1

# Copy specific files needed for building
COPY requirements.txt .

# Install build dependencies, Python packages, and preload models
RUN apt-get update && apt-get install -y --no-install-recommends \
    libffi-dev \
    libssl-dev \
    libsm6 \
    libxext6 \
    ffmpeg \
    imagemagick \
    libmagickwand-dev \
    gcc \
    libc-dev \
    libkrb5-dev \
    libsasl2-dev \
    python3-dev \
    cmake \
    build-essential && \
pip install --no-cache-dir -r requirements.txt && \
python -c "import nltk; nltk.download('vader_lexicon')" && \
python -c "import whisper; whisper.load_model('tiny.en'); whisper.load_model('medium.en')" && \
python -c "from retinaface import RetinaFace; RetinaFace.build_model()" && \
apt-get remove --purge -y gcc libc-dev libkrb5-dev libsasl2-dev python3-dev cmake build-essential && \
apt-get autoremove -y && \
rm -rf /var/lib/apt/lists/*

# Modify ImageMagick policy for the container environment
RUN sed -i \
-e '/<!-- in order to avoid to get image with password text -->/,+1d' \
-e '/disable ghostscript format types/,+6d' \
/etc/ImageMagick-6/policy.xml

# Change permissions of tmp directory to ensure accessibility
RUN chmod 1777 /tmp

# Copy the application files
COPY app.py best_clips.py deploy.prototxt res10_300x300_ssd_iter_140000.caffemodel Montserrat-Black.ttf ./

# Expose the Flask
EXPOSE 5000

# Change to nonroot system user
USER nonroot

# Run app.py when the container launches
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]