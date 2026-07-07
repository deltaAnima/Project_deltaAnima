#!/bin/bash

source ./.venv/bin/activate

# force main server to use 4 - 7 core
taskset -c 4-9 uvicorn Reminh_main:app --host 0.0.0.0 --port 36722
