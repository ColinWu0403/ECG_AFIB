# Use the official Python base image with the specified version
FROM python:3.11.7

# Set the working directory in the container
WORKDIR /

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install optuna-integration

# Copy the rest of the application code to the working directory
COPY . /

# Specify the command to run the application
CMD ["python", "src/processECG.py"]

# docker run -v ${pwd}/data/:/data -it --rm ecg_ml