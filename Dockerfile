# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy all files from your repo into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the API port
EXPOSE 7860

# Start the Flask app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
