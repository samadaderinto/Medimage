# Use an official Ubuntu base image
FROM ubuntu:20.04

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Update, install necessary packages with no-recommends, and clean up in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends clang++ build-essential curl libcurl4-openssl-dev libssl-dev cmake && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory to /app
WORKDIR /app

# Copy the entire project into the container
COPY . .

# Create a build directory and compile the project,
# ensuring that the executable is built from server.cpp
RUN mkdir -p build && \
    clang++ -std=c++17 -I./include -o build/server server.cpp -lcurl

# Set the default command to run the executable from the build directory
CMD ["./build/server"]
