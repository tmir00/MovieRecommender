#!/bin/bash

# Stop the running db container
docker-compose stop db

# Remove the existing db container
docker-compose rm -f db

# Remove the existing named volume
docker volume rm movierecommender_dbdata

# Re-create and start the container, which will reinitialize the database
docker-compose up -d db
