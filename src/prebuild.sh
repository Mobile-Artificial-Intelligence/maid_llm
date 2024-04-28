#!/bin/bash

# Define the base directory to start searching from
BASE_DIR="/usr"

# Find the OpenBLAS library directory
LIB_DIR=$(sudo find /usr/lib /usr/lib64 -name "libopenblas*" -type f | head -1 | xargs dirname)

# Find the OpenBLAS include directory
INCLUDE_DIR=$(find $BASE_DIR -type d -name "openblas" | head -1)

# Error checking
if [ -z "$LIB_DIR" ]; then
    echo "Error: Could not find OpenBLAS library directory"
    exit 1
fi

if [ -z "$INCLUDE_DIR" ]; then
    echo "Error: Could not find OpenBLAS include directory"
    exit 1
fi

# Set the install prefix (customize this as needed)
INSTALL_PREFIX=$BASE_DIR

# Define the version (customize this as needed)
VERSION="0.3.21"

# Create the .pc file from the template
sed "s|@CMAKE_INSTALL_PREFIX@|$INSTALL_PREFIX|g;
     s|@OPENBLAS_LIBRARY_DIR@|$LIB_DIR|g;
     s|@OPENBLAS_INCLUDE_DIR@|$INCLUDE_DIR|g;
     s|0.3.21|$VERSION|g" openblas.pc.in > openblas.pc
