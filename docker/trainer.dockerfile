# Base image
FROM python:3.10-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Lets copy over our application (the essential parts) from our computer to the container:
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY models/ models/

# Lets set the working directory in our container and add commands that install the dependencies:
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

# Finally, we are going to name our training script as the entrypoint for our docker image. 
# The entrypoint is the application that we want to run when the image is being executed:
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
