#!/bin/bash

# Remove Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +

# Remove log files
find . -type f -name "*.log" -delete

# Other cleanup tasks as needed
