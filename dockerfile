# Use official Python image from the Docker Hub
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . .

# Expose the Streamlit port (8501 by default)
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "ui.py", "--server.enableCORS=false"]
