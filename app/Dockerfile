
# https://cloud.google.com/deep-learning-containers/docs/choosing-container#pytorch_versions
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-13.py310

WORKDIR /app

COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

# CMD ["bash"]
    
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $AIP_HTTP_PORT"]
