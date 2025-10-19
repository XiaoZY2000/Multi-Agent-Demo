#!/bin/bash

# Use sudo docker if docker is not running as a non-root user
DOCKER_CMD="docker"
if ! docker ps >/dev/null 2>&1; then
  DOCKER_CMD="sudo docker"
fi


# Set container names and ports
CONTAINER_NAME_1="ollama-phi4"
CONTAINER_NAME_2="ollama-phi4-sub"
HOST_PORT=11434
CONTAINER_PORT=11435
MODEL_NAME="phi4-mini-reasoning"

# Pull the Ollama Docker image
echo "Pulling Ollama Docker image..."
$DOCKER_CMD pull ollama/ollama

# Stop and remove any existing containers with the same name
if $DOCKER_CMD ps -a --format '{{.Names}}' | grep -q "$CONTAINER_NAME"; then
  echo "Removing old container $CONTAINER_NAME..."
  $DOCKER_CMD rm -f "$CONTAINER_NAME"
fi

# Start a new Ollama container with the specified name and port
echo "Starting new Ollama container: $CONTAINER_NAME_1"
$DOCKER_CMD run -d \
  --name "$CONTAINER_NAME_1" \
  -p "$HOST_PORT":11434 \
  -v ollama-data-phi4-mini-reasoning:/root/.ollama \
  ollama/ollama

# Pull the specified model into the Ollama container
echo "Pulling model: $MODEL_NAME..."
$DOCKER_CMD exec -it "$CONTAINER_NAME_1" ollama pull "$MODEL_NAME"

echo "Ollama server is running on port $HOST_PORT with model '$MODEL_NAME'"

echo "Starting new Ollama container: $CONTAINER_NAME_2"

# Start a second Ollama container with a different name and port
$DOCKER_CMD run -d \
  --name "$CONTAINER_NAME_2" \
  -p "$CONTAINER_PORT":11434 \
  -v ollama-data-phi4-mini-reasoning-sub:/root/.ollama \
  ollama/ollama

# Pull the second model into the second Ollama container
echo "Pulling model: $MODEL_NAME..."
$DOCKER_CMD exec -it "$CONTAINER_NAME_2" ollama pull "$MODEL_NAME"
echo "Ollama server is running on port $CONTAINER_PORT with model '$MODEL_NAME'"