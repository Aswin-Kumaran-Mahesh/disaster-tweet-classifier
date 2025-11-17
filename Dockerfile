FROM python:3.11-slim

# Work inside /app
WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

# Expose the port Spaces expects for Docker apps
EXPOSE 7860

# Run our Streamlit app (NOT src/streamlit_app.py)
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]