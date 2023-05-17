FROM python:3.10

# install git iproute2
RUN apt-get update && apt-get -y install git iproute2

RUN pip install --no-cache-dir \
    autopep8 \
    flake8
# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
