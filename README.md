# Video_Classification_UCF101
Video Classification on UCF101 dataset


# Preprocessing: The old fashioned way

First, download the dataset from UCF Repository [https://www.crcv.ucf.edu/data/UCF101.php] then run the UCF_Preprocessing.py file to preprocess UCF101 dataset.

In the preprocessing phase we used a different technique in which we extracted exactly 1,650 frames per category
meaning 1,650 x 101 = 1,66,650 frames or you can say images in whole dataset

# About Dataset

UCF101 folder:
~100 to 150 videos per category

~13,320 total videos

after combining all the videos of each and every single category and saving them in dataset named folder
dataset folder:

~only 1 video per category

~101 total videos in dataset folder after combining all the videos

training folder:

~1500 frames per category

~1500 x 101 = 151500 total frames

How do we achieve this, you may ask, well it's not that easy as it seems

What we did was calculated total number of frames of every video of single category
meaning: 
	let's say ApplyEyeMakeup has 10 videos of 5 seconds long clips each
	and let's say that those 10 videos are of 30 FPS (Frames per second)
	number of frames in one category = number of videos in each category x length of a clip x frame rate of a single video = 10 x 5 x 30 = 1500 total number of frames in a SINGLE CATEGORY
	and let's assume we only need 750 frames
	so we take every second frame and write it out onto the "training_set/" folder present on your same working directory
	that's how we separated frame from the videos and made a BALANCED DATASET

testing folder:

~150 frames per category

~150 x 101 = 15150 total frames

In order to make our testing data, we randomly selected 150 frames from training set for each category and moving them from "training_set/" and storing them onto "testing_set/" named folder



frames in training set = 151500
frames validation set = frames taken from validation set = 20% of training set (30300 frames)
frames in testing set = 15150 (10% of training set)



for training purpose we used "training_set/" directory and for testing we used "testing_set/" directory


# Model Analysis:
models used:

	ResNet50
	
	ResNet101
	
	ResNet50V2
	
	ResNet101
	
	MobileNet
	
	MobileNetV2

MobileNet and MobileNetV2 are worst model to perform video classification because they aren't made for heavy datasets infact they are made for Mobile and Embedded Devices, hence named "Mobile"
also MobileNets are giving good accuracies but have higher losses, that's why we discarded this model

ResNet50 and ResNet50v2, both are giving much impressive results than their counterparts MobileNets but took much time for training because of the fact that it contains more deeper hidden layers than MobileNet models.




# Required Parameters
dataset = "UCF-101/"                	  		# Actual Dataset Path

dataset2 = "dataset/"               	  		# After Combining all videos of the Dataset, the recreated Dataset Path

train_path = "training_set/"     	      		# Training Path

test_path = "testing_set/"          	  		# Testing Path

no_of_frames = 1650                 	  		# Number of Frames to be extracted from a single category

epochs = 20                                     	# Number of epochs to run

batch_size = 32                	 			# Batch Size

n_classes = 101                	  			# Number of Classes

optimizer = "Adam" 					# Adam (adaptive momentum) optimizer is used

loss_metric = "categorical_crossentropy"  		# Loss Metric used for every model is one and same

last_layer_activation_function = "softmax"		# Softmax function is used for last layer


input shape of ResNet50, ResNet101, ResNet50V2, ResNet101, MobileNet and MobileNetV2 are all the same and that is: (224, 224, 3) => [image height, image width and number of channels]
