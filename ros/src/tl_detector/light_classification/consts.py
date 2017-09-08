# Author: Guy Hadash
# Source: https://github.com/mynameisguy/TrafficLightChallenge-DeepLearning-Nexar

## dirs confs
DATASET_FOLDER="./dataset"
MODELS_CHECKPOINTS_DIR="./checkpoints"

# training confs
BATCH_SIZE = 64
TRAINING_EPOCHS = 200 #max
TRAIN_IMAGES_PER_EPOCH = 16768
VALIDATE_IMAGES_PER_EPOCH = 1856
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
