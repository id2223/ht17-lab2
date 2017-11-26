#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR && \

docker rm -f ts
docker run -p 8888:8888 -v $DIR/notebooks:/notebooks --name ts -it gcr.io/tensorflow/tensorflow
