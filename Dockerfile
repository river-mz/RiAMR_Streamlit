# Use Python 3.11 slim image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libssl-dev \
        libxml2-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /srv/streamlit-app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy app scripts and resources
COPY /streamlit_running ./

# Set permissions
RUN chmod -R 755 /srv/streamlit-app/

# Expose Streamlit port
EXPOSE 8501

# Optional log directory
VOLUME ["/var/log"]

# Start Streamlit app
CMD ["streamlit", "run", "streamlit_website_updated.py", "--server.port=8501", "--server.address=0.0.0.0"]