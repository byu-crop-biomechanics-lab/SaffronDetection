#!/bin/bash -ex
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

$DIR/bootstrap.sh $DIR $DIR/venv

$DIR/venv/bin/python $DIR/src/main.py $@ --camera-port 50052 --canbus-port 50060 --stream-every-n 1

exit 0
