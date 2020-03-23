# Importing Essential Libraries
import os
# import cv2
import time
# import math
# import glob
# import random
# import tensorflow
# import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, EarlyStopping

# Required Parameters
dataset = "UCF-101/"                                                            # Dataset Path
dataset2 = "dataset/"                                                           # Dataset2 Path
train_path = "training_set/"                                                    # Training Path
test_path = "testing_set/"                                                      # Testing Path
no_of_frames = 1650                                                             # Number of Frames
ch = 4                                                                          # Model Selection Choice
epochs = 20                                                                     # Number of epochs
batch_size = 32                                                                 # Batch Size
n_classes = 101                                                                 # Number of Classes
patience = 2                                                                    # Patience for EarlyStopping
stime = int(time.time())                                                        # Defining Starting Time
categories = os.listdir(dataset)                                                # Name of each Class/Category

# For Kaggle Purpose
# train_path = "/kaggle/input/vid-classification-ucf101/UCF/training_set/"        # Training Path for Kaggle
# test_path = "/kaggle/input/vid-classification-ucf101/UCF/testing_set/"          # Testing Path for Kaggle

# categories.sort()
# print(categories)

# Defining ResNet Architecture
# resnet = tensorflow.keras.applications.resnet_v2.ResNet50V2()

# Defining Base Model
if ch == 1:
    from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
    base_model = ResNet50(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))
elif ch == 2:
    from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
    base_model = ResNet101(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))
elif ch == 3:
    from tensorflow.keras.applications.resnet import ResNet150, preprocess_input
    base_model = ResNet150(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))
elif ch == 4:
    from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
    base_model = ResNet50V2(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))
elif ch == 5:
    from tensorflow.keras.applications.resnet_v2 import ResNet101V2, preprocess_input
    base_model = ResNet101V2(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))
elif ch == 6:
    from tensorflow.keras.applications.resnet_v2 import ResNet150V2, preprocess_input
    base_model = ResNet150V2(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))
elif ch == 7:
    from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
    base_model = MobileNet(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))
elif ch == 8:
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
    base_model = MobileNetV2(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation = 'relu')(x)
x = Dropout(0.5)(x)
# x = Dense(512, activation = 'relu')(x)
# x = Dense(256, activation = 'relu')(x)
preds = Dense(n_classes, activation = 'softmax')(x)

model = Model(inputs = base_model.input, outputs = preds)

# Printing the names of each layer
for i, layer in enumerate(model.layers):
    print(i, layer.name)

# Setting each layer as trainable
for layer in model.layers:
    layer.trainable = True

# Setting 1/3 layers as trainable
# for layer in model.layers[:65]:
#     layer.trainable = False
# for layer in model.layers[65:]:
#     layer.trainable = True

# Defining Image Data Generators
train_datagenerator = ImageDataGenerator(preprocessing_function = preprocess_input,
                                         validation_split = 0.2)

test_datagenerator = ImageDataGenerator(preprocessing_function = preprocess_input)

train_generator = train_datagenerator.flow_from_directory(train_path,
                                                          target_size = (224, 224),
                                                          color_mode = 'rgb',
                                                          batch_size = batch_size,
                                                          class_mode = 'categorical',
                                                          shuffle = True)

validation_generator = train_datagenerator.flow_from_directory(train_path,
                                                               target_size = (224, 224),
                                                               color_mode = 'rgb',
                                                               batch_size = batch_size,
                                                               class_mode = 'categorical',
                                                               subset = 'validation')

test_generator = test_datagenerator.flow_from_directory(test_path,
                                                        target_size = (224, 224),
                                                        color_mode = 'rgb',
                                                        class_mode = 'categorical')

print(train_generator.class_indices)
print(validation_generator.class_indices)
print(test_generator.class_indices)

# Compiling the Model
model.compile(optimizer = "Adam",
              loss = "categorical_crossentropy",
              metrics = ["accuracy"])

# Creating a timestamp directory
try:
    os.mkdir("{}_{}b_{}e".format(stime, batch_size, epochs))
except:
    print("Directory already present...")

# CSVLogger
filename = "{}_{}b_{}\\file.csv".format(stime, batch_size, epochs)
csv_log = CSVLogger(filename)

# Early Stopping
# early_stopping = EarlyStopping(patience = patience)

# Tensorboard
tensorboard = TensorBoard(log_dir = "{}_{}b_{}e\logs".format(stime, batch_size, epochs))

# Defining Model Checkpoint
checkpoint_name = "{}_{}b_{}e".format(stime, batch_size, epochs)
checkpoint_path = checkpoint_name + "\cp-{epoch:04d}-{accuracy:.4f}a-{loss:.4f}l-{val_accuracy:.4f}va-{val_loss:.4f}vl.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
modelcheckpoint = ModelCheckpoint(checkpoint_path)

# Training the Model
history = model.fit(train_generator,
                    validation_data = validation_generator,
                    epochs = epochs,
                    callbacks = [modelcheckpoint, tensorboard, csv_log])

# Plotting the Graph
model_history = pd.DataFrame(history.history)
model_history.plot()

# Loading Model
from tensorflow.keras.models import load_model
model = load_model(r"1584534392_32b_20e/cp-0002-0.9593a-0.1368l-0.9570va-0.1419vl.h5")

# Evaluating Model's Performance
history2 = model.evaluate_generator(test_generator)
# history2 = model.evaluate(test_generator)