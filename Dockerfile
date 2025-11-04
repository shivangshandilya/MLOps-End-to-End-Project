FROM python:3.10-slim

RUN pip install --upgrade pip

WORKDIR /app

COPY . /app

# Install requirements
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8080

# Run the app when container starts
CMD ["python", "app.py"]