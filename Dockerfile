
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-13.py310
# FROM gcr.io/deeplearning-platform-release/pytorch-latest-cpu-v20230501
# FROM gcr.io/deeplearning-platform-release/tf-gpu.1-13
# FROM gcr.io/deeplearning-platform-release/pytorch.1-13
# FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest

# FROM python:3.11.1

WORKDIR /app

# COPY ./app/requirements.txt /requirements.txt
# RUN pip install -r /requirements.txt

COPY ./app /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

# CMD ["bash"]
    
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $AIP_HTTP_PORT"]
