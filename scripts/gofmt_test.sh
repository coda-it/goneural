#!/bin/bash

if [ -n "$(gofmt -l . | grep -v vendor/)" ]; then
    echo "Go code is not formatted in the following files:"
    gofmt -l .
    exit 1
fi
