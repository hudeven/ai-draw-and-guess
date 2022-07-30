#!/bin/bash

ZIPFILE="model_and_pretrained.zip"
MODELSDIR="models"
PRETRAINED="pretrained"


# zip source code and pretrained checkpoints
# zip -r $ZIPFILE $MODELSDIR $PRETRAINED 

# Create *.mar
torch-model-archiver --model-name dalle_mega \
                     --version 1.0 \
                     --handler handler.py \
                     --extra-files $ZIPFILE \
                     --force
