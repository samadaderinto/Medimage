# Backend
FROM ubuntu:22.04 AS backend

# Avoid timezone prompt during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install required packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libcurl4-openssl-dev \
    git \
    libboost-all-dev \
    python3-pip \
    libasio-dev \
    pkg-config \
    nlohmann-json3-dev \
    libopencv-dev \
    wget \
    unzip

# Download and build Crow
WORKDIR /tmp
RUN git clone https://github.com/CrowCpp/Crow.git && \
    cd Crow && \
    mkdir build && \
    cd build && \
    cmake .. -DCROW_BUILD_EXAMPLES=OFF -DCROW_BUILD_TESTS=OFF && \
    cmake --build . && \
    cmake --install .

# Download and install ONNX Runtime
WORKDIR /tmp
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-linux-aarch64-1.15.1.tgz && \
    tar xvf onnxruntime-linux-aarch64-1.15.1.tgz && \
    mkdir -p /backend/external/onnxruntime && \
    cp -r onnxruntime-linux-aarch64-1.15.1/* /backend/external/onnxruntime/ && \
    rm -rf onnxruntime-linux-aarch64-1.15.1*

# Set working directory for our application
WORKDIR /backend

# Copy the entire backend directory
COPY . .

# Debug: list include dir to verify Crow headers are present
RUN echo "Listing /backend/include:" && ls -la /backend/include || true
RUN echo "Find files under /backend/include:" && find /backend/include -maxdepth 3 -type f -print || true

# Create build directory and build the project
RUN mkdir -p build && \
    cd build && \
    cmake .. && \
    cmake --build .

# Run the server
CMD ["./build/Medimage"]


