import os
import cv2
from .siamese import build_model
from .dataloader import PairsDataLoader, DataLoader
from .utils import create_anchor_images, plot_anchor_images
from .utils import create_prediction_pairs, concat_pair, plot_pairs
from .detectface import detect_face


print(f"CURRENT DIRECTORY: {os.getcwd()}\n")


## Build model
model = build_model()
model.load_weights('project\siamese_100x100_v1_gpu\checkpoint_2.h5')
model.trainable = False

#Create train/val data
BATCH_SIZE_TARGET = {'Train': 500, 'Val': 500}
IMAGE_SIZE = (100, 100)
print("\nloading training val data...\n")
(Train), (Val) = PairsDataLoader('DetectedImages', IMAGE_SIZE, BATCH_SIZE_TARGET).run()

#create anchor images
DIRS = {name: os.listdir(f'DetectedImages\{name}')[:32] for name in ['Drake', 'Eminem', 'Messi', 'Ronaldo']}
ANCHOR_DRAKE = DIRS['Drake']
ANCHOR_MESSI = DIRS['Messi']
ANCHOR_EMINEM = DIRS['Eminem']
ANCHOR_RONALDO = DIRS['Ronaldo']
ANCHORS = {'Drake': ANCHOR_DRAKE, 'Messi': ANCHOR_MESSI, 'Eminem': ANCHOR_EMINEM, 'Ronaldo': ANCHOR_RONALDO}
print("\nloading anchor images...\n")
FACES = DataLoader('DetectedImages').run()
anchor_images = create_anchor_images(ANCHORS, FACES)

test_image = cv2.imread('images\drake3.jpeg')





