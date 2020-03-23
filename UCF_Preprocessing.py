# Importing Essential Libraries
import os
import cv2
import glob
import random
from tqdm import tqdm

# Required Parameters
dataset = "UCF-101/"                    # Dataset Path
dataset2 = "dataset/"                   # Dataset2 Path
train_path = "training_set/"            # Training Path
test_path = "testing_set/"              # Testing Path
no_of_frames = 1650                     # Number of Frames
categories = os.listdir(dataset)        # Name of each Class/Category

# Creating dataset directory
try:
    os.mkdir(dataset2)
    print("Folder {} created...".format(dataset2))
except:
    print("A folder {} already exists...".format(dataset2))

# Creating training_set directory
try:
    os.mkdir(train_path)
    print("Folder {} created...".format(train_path))
except:
    print("A folder {} already exists...".format(train_path))

# Creating training_set directory
try:
    os.mkdir(train_path)
    print("Folder {} created...".format(train_path))
except:
    print("A folder {} already exists...".format(train_path))

# Creating testing_set directory
try:
    os.mkdir(test_path)
    print("Folder {} created...".format(test_path))
except:
    print("A folder {} already exists...".format(test_path))

# Creating same directories for dataset2/ that are already present in the dataset directory
for category in categories:
    try:
        os.mkdir(dataset2 + category)
        print("Folder {} created...".format(dataset2))
    except:
        print("A folder already exists, named {}...".format(category, dataset))

# Creating same directories for training_set/ that are already present in the dataset directory
for category in categories:
    try:
        os.mkdir(train_path + category)
        print("Folder {} created...".format(category))
    except:
        print("A folder already exists, named {}...".format(category, train_path))

# Creating same directories for testing_set/ that are already present in the dataset directory
for category in categories:
    try:
        os.mkdir(test_path  + category)
        print("Folder {} created...".format(category))
    except:
        print("A folder already exists, named {}...".format(category, test_path))

# Combining multiple videos into single video file
for category in tqdm(categories):
    videofiles = [dataset + category + "/" + n for n in os.listdir(dataset + category) if n[-4:]==".avi"]
    video_index = 0
    cap = cv2.VideoCapture(videofiles[0])    
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter("{}/{}/{}.avi".format(dataset2, category, category), fourcc, 25, (320, 240))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            video_index += 1
            if video_index >= len(videofiles):
                break
            else:
                cap = cv2.VideoCapture(videofiles[ video_index ])
                ret, frame = cap.read()
                out.write(frame)
        else:
            out.write(frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Getting total no. of frames in each classes
total_frames = []
for category in tqdm(categories):
    cap = cv2.VideoCapture(dataset2 + category + "/" + category + ".avi")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames.append(length)
    cap.release()
    cv2.destroyAllWindows()

# Extracting 1650 images from each category
for category in tqdm(categories):
    a = glob.glob(dataset2 + category + '/*.avi')
    for i in range(len(a)):
        count = 0
        cap = cv2.VideoCapture(a[0])
        while(cap.isOpened()):
            frame_id = cap.get(1)
            ret, frame = cap.read()
            if ret != True:
                print("Exiting...")
                break
            if frame_id % int(total_frames[categories.index(category)] / no_of_frames) == 0.0:
                if count >= no_of_frames:
                    break
                file_name = train_path + category + '/frame_{}.jpg'.format(count); count += 1;
                cv2.imwrite(file_name, frame)
        cap.release()
        cv2.destroyAllWindows()

# Extracting one frame per five frame from the Videos
# for category in tqdm(categories):
#     count = 0    
#     a = glob.glob(dataset + '/' + category + '/*.avi')
#     for i in range(len(a)):
#         cap = cv2.VideoCapture(a[i])
#         frameRate = cap.get(5)
#         while(cap.isOpened()):
#             frameId = cap.get(1)
#             ret, frame = cap.read()
#             if (ret != True):
#                 break
#             if (frameId % math.floor(frameRate) == 0):
#                 cv2.imwrite(train_path + '/' + category + '/{}_{}.jpg'.format(category, count), frame)
#                 count += 1
#         cap.release()

# Extracting every frame from the Videos
# for category in tqdm(categories):
#     count = 0    
#     a = glob.glob(dataset + category + '/*.avi')
#     for i in range(len(a)):
#         cap = cv2.VideoCapture(a[i])
#         # frameRate = cap.get(5)
#         while(cap.isOpened()):
#             # frameId = cap.get(1)
#             ret, frame = cap.read()
#             if (ret != True):
#                 break
#             # if (frameId % math.floor(frameRate) == 0):
#             else:
#                 cv2.imwrite(train_path + category + '/{}_{}.jpg'.format(category, count), frame)
#                 count += 1
#         cap.release()

# Moving random images from training_set into testing_set
for category in tqdm(categories):
    sub_file = [file for file in glob.glob(train_path + category + "\*")]
    test_files = random.sample(sub_file, 150)
    for test_file in test_files:
        img = cv2.imread(test_file)
        os.remove(test_file)
        cv2.imwrite(test_path + category + '/' + test_file.split("\\")[-1] , img)

# Counting number of images in each folder of training set
for category in categories:
    print(len(os.listdir(train_path + category)), "in training &",
          len(os.listdir(test_path + category)), "in testing", ":", category)