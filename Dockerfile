# Use an official minimal image with Linux
FROM alpine:latest

# Install dependencies
RUN apk add --no-cache curl

# Download and install PocketBase
RUN curl -L -o /pocketbase https://github.com/pocketbase/pocketbase/releases/latest/download/pocketbase-linux-amd64
RUN chmod +x /pocketbase

# Expose the PocketBase default port
EXPOSE 8090

# Run PocketBase
CMD ["/pocketbase", "serve", "--http", "0.0.0.0:8090"]
