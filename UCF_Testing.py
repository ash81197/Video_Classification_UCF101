# Importing Essential Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from sklearn.metrics import classification_report,precision_score,recall_score,f1_score,confusion_matrix

# Required Parameters
test_path = "testing_set/"

# Loading Tensorflow.Keras Model
bring_model = tf.keras.models.load_model("1584534392_32b_20e/cp-0003-0.9770a-0.0768l-0.9739va-0.0973vl.h5")

# Image Data Generator
test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

test_generator = test_datagen.flow_from_directory(test_path,
                                                  target_size = (224, 224),
                                                  color_mode = "rgb",
                                                  shuffle = False,
                                                  class_mode = 'categorical',
                                                  batch_size = 1)

activities = test_generator.class_indices
print(activities)

def get_activity(val):
    for key, value in activities.items():
        if val == value:
            return key
    return "Invalid"

filenames = test_generator.filenames
nb_samples = len(filenames)

# Prediction
predict = bring_model.predict_generator(test_generator, steps = nb_samples, verbose = 1)

y_pred = []
for val in predict:
    y_pred.append(get_activity(np.argmax(val)))

y_true = []
for file in filenames:
    y_true.append(file.split("\\")[0])

cm = confusion_matrix(y_true,y_pred)

print(precision_score(y_true, y_pred, average = 'macro'))
print(recall_score(y_true, y_pred, average = 'macro'))
print(f1_score(y_true, y_pred, average = 'macro'))

print(precision_score(y_true, y_pred, average = 'micro'))
print(recall_score(y_true, y_pred, average = 'micro'))
print(f1_score(y_true, y_pred, average = 'micro'))

# Making a Classification Report
print(classification_report(y_true, y_pred))

dataframe = pd.DataFrame(cm)
inv_dict = {v: k for k, v in activities.items()} 
dataframe = dataframe.rename(index = inv_dict)
dataframe = dataframe.rename(columns = inv_dict)

# Saving Confusion Matrix to the disk in CSV format
dataframe.to_csv("Perfomance Confusion Matrix.csv")