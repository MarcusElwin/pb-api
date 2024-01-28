CONTAINER_REGISTRY:="local-dev"
VERSION:="latest"
TAG:=$CONTAINER_REGISTRY/pb-api:$VERSION
DOCKERFILE:=Dockerfile

.PHONY: build # build image using docker-compose with tag
build:
	@echo "Building images with tag: $TAG"
	docker-compose build

.PHONY: up # run container
up:
	@echo "Running images with tag: $TAG"
	docker-compose up --build

.PHONY: run # run container in interactive mode using docker-compose with tag
run:
	@echo "Running container with tag: $TAG"
	docker-compose run --rm pb-rest-api

.PHONY: down # stop container using docker-compose
down:
	@echo "Stopping container"
	docker-compose down

.PHONY: destroy # destroy container using docker-compose	
destroy:
	@echo "Destroying container"
	docker-compose down --rmi all

.PHONY: train # run training script using docker-compose with tag
train:
	@echo "Running training script"
	docker-compose run --rm pb-train

.PHONY: logs # show logs using docker-compose
logs:
	@echo "Showing logs"
	docker-compose logs pb-rest-api

.PHONY: help # shows available commands
help:
	@echo "\nAvailable commands:\n\n $(shell sed -n 's/^.PHONY:\(.*\)/ *\1\\n/p' Makefile)"
