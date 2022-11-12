#!/usr/bin/env sh

CONTAINER_REGISTRY="local-dev"
VERSION="latest"
TAG=$CONTAINER_REGISTRY/pb-api:$VERSION
export DOCKERFILE=Dockerfile

# build images
echo "Building images with tag: $TAG"

docker build --tag $TAG -f $(dirname $0)/$DOCKERFILE $(dirname $0)

# run container in interactive mode
echo "Running container with tag: $TAG"
docker run --rm -it --entrypoint /bin/bash $TAG
