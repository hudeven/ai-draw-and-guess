#!/bin/bash

ZIPFILE="model_and_pretrained.zip"
MODELSDIR="models"
PRETRAINED="pretrained"
# zip source code and pretrained checkpoints
# zip -r $ZIPFILE $MODELSDIR

# Create *.mar
torch-model-archiver --model-name dalle_image_gen \
                     --version 1.0 \
                     --handler handler.py \
                     --extra-files $ZIPFILE \
                     --force
