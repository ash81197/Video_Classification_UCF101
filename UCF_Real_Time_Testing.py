# Importing Essential Libraries
import os
import cv2
import time
import numpy as np

# Data Directory
dataset = 'UCF-101/'

# Saved Model Name
# modelname = 'SLD_0220_2_mobilenet.h5'
# modelname = 'Saved Models/SLD_0208.h5'
# modelname = r"1583918311_64b_2e\checkpoints_1583918311_64b_2e\cp-0002.h5"
# modelname = "UCF_0304_2.h5"
# modelname = r"1583984942_64b_100e\checkpoints\cp-0001.h5"
modelname = r"1584534392_32b_20e\cp-0002-0.9593a-0.1368l-0.9570va-0.1419vl.h5"

# Categories of y-variable
categories = os.listdir(dataset)

# Loading Saved Model
from tensorflow.keras.models import load_model
model = load_model(modelname)#, compile=False)
#model = load_model("SLD_0208.h5")

# Preparing the predicting image
#def prepare(file):
#    IMG_SIZE = 224
#    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
#    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#def load_image(file, show = False):
#    img = cv2.imread(file)
#    # cv2.imshow('image', img)s
#    img = cv2.resize(img, (224, 224))
#    img = np.expand_dims(img, axis = 0)
#    img = img / 255.
#    if show:
#        plt.imshow(img[0])
#        plt.axis('off')
#        plt.show()
#    return img

# Enabling WebCam to Capture Images
capture = cv2.VideoCapture("test/basketball.mpg")

# Actual Processing and Prediction of the image until camera is ON using OpenCV
while True:
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        try:
            img_size = 224

            cropped_img = frame#[0:224, 0:224]
            img_array = cropped_img
            img_array = cv2.resize(img_array, (img_size, img_size))
            img_array = np.expand_dims(img_array, axis = 0)
            img_array = img_array / 255.
                        
            # cv2.imshow('frame', frame)
            # cv2.imshow('Cropped Img', cropped_img)
            
            prediction = model.predict([img_array])
            # print(categories[np.argmax(prediction)])
            label = categories[np.argmax(prediction)]
            cv2.putText(frame,
                        label,
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 0),
                        2)
            cv2.imshow('Video Classification', frame)            
            # print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        except:
            pass
    # When 'Q' is presses the pop-up will exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Destroying the Session
capture.release()
cv2.destroyAllWindows()