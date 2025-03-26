# Use an official minimal image with Linux
FROM debian:latest

# Install dependencies
RUN apt update && apt install -y curl

# Download and install the correct PocketBase binary for x86_64
RUN curl -L -o /pocketbase https://github.com/pocketbase/pocketbase/releases/latest/download/pocketbase-linux-x86_64
RUN chmod +x /pocketbase

# Expose PocketBase default port
EXPOSE 8090

# Run PocketBase
CMD ["/pocketbase", "serve", "--http", "0.0.0.0:8090"]
