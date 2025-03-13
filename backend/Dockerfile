FROM ubuntu:22.04

# Install dependencies
RUN apt update && apt install -y \
    clang \
    cmake \
    git \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Create build directory, compile project
RUN mkdir build && cd build && cmake .. && make

# Create upload directory
RUN mkdir -p /app/uploads

# Expose server port
EXPOSE 8080

# Run the server
CMD ["./build/server"]
