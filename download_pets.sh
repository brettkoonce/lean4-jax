#!/bin/bash
set -e
mkdir -p data/pets
cd data/pets
if [ ! -f "train.bin" ]; then
  if [ ! -d "images" ]; then
    echo "Downloading Oxford-IIIT Pets images..."
    curl -LO https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
    echo "Extracting images..."
    tar xzf images.tar.gz
    rm images.tar.gz
  fi
  if [ ! -d "annotations" ]; then
    echo "Downloading annotations..."
    curl -LO https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
    echo "Extracting annotations..."
    tar xzf annotations.tar.gz
    rm annotations.tar.gz
  fi
  echo "Preprocessing to binary format (requires: pip install Pillow)..."
  python3 ../../preprocess_pets.py . .
fi
echo "Done. Files in ./data/pets/"
ls -lh *.bin
