# Introduction

This project is done as a part of the Nanodegree - Self-Driving Car Engineer provided by Udacity. The scope of this project is finding and marking vehicles on a given, recorded track. The recorded raw video and some used sample codes are provided by Udacity.

The goals / steps of this project are the following:

- Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
- Optionally, apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector
- Normalize features and randomize a selection for training and testing
- Implement a sliding-window technique and use the trained classifier to search for vehicles in images
- Apply the the pipeline on a video stream and create a heat map of recurring detections frame by frame 
- Estimate a bounding box for detected vehicles

# Outline
1. Requirements
2. Files
3. 
Pipeline for Single Images
Pipeline for Video
Conclusion

# 1. Requirements
- Python 3.5
- Environment [CarND-Term1-Starter-Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) provided by Udacity
- [Project Data](https://github.com/udacity/CarND-Vehicle-Detection) provided by Udacity
- Labeled images of [vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip)
- Labeled images of [non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)
- Some code is taken out the material of the lectures of the Nanodegree *Self-Driving Car Engineer* provided by Udacity

# 2. Files
- README.md - Explains the structure of the software and the approach to solve the problem
- P5.ipynb - Implementation of helper functions and pipeline
- test_video.mp4 - Short raw video for testing the pipeline
- project_video.mp4 - Raw video which has to be handled properly to pass the project review
- CarND-Vehicle-Detection - project video.mp4 - Output for project_video.mp4 after using the pipeline
- folder info_for_readme - images, which are included in this readme
- folder test_images - test images provided by Udacity

# 3. Classifier
### 3.1 Collect Images for Training and Test Data
First, all the images of vehicle and non-vehicle for training and testing the classifier have to be read in. 

The code for this functionality can be found in **CODE CELL 2** of the Jupyter notebook P5.jpynb.
Functions `get_car_data` and `get_non_car_data`.

Labeled images of [vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and of [non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) have been taken. These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.

All the images are 64 x 64 pix and look like the following examples:
![car_non_car](https://github.com/gada1982/CarND-Vehicle-Detection/blob/master/info_for_readme/car_non_car.png)

The implementation of the classifier uses two different types of features to improve the result. Histograms of Color in combination with Spatial Binning of Color as the first and Histogram of Oriented Gradient (HOG) features as the second.

### 3.2 Histograms of Color / Spatial Binning of Color
The code for this functionality can be found in **CODE CELL 4** of the Jupyter notebook P5.jpynb.
Functions `get_parameters_color_features`, `bin_spatial` and `color_hist`.

I have experimented with different parameters. The best result (feature size and accurancy while training) was produced by the following parameters: 

`cspace = 'YCrCb', spatial_size = (32,32), hist_bins = 32, hist_range = (0, 256)`

`np.histogram` has been used for creating the histogram for a single channel.

### 3.3 Histogram of Oriented Gradient (HOG)
The code for this functionality can be found in **CODE CELL 5** of the Jupyter notebook P5.jpynb.
Functions `get_parameters_hog_features`, `get_hog_features` and `print_hog_examples`.

I have experimented with different parameters. The best result (feature size and accurancy while training) was produced by the following parameters: 

`cspace = 'YCrCb', orient = 9, pix_per_cell = 8, cell_per_block = 2, hog_channel = 'ALL'`

The functionality `hog from skimage.feature` has been used for creating the Histogram of Oriented Gradient (HOG).

The following example shows the result for one car-image and one non-car image:
![hog](https://github.com/gada1982/CarND-Vehicle-Detection/blob/master/info_for_readme/hog.png)

### 3.4 Extracting features for Training
The code for this functionality can be found in **CODE CELL 6** of the Jupyter notebook P5.jpynb.
Functions `extract_features`,  `extract_car_features` and `extract_non_car_features`.

Features for cars and non-cars have been extracted seperately, but for both the same base function `extract_features` has been used.

### 3.5 Create Training Data
Out of the extracted features for cars and non-cars training data for the classifier has been generated. The code for this functionality can be found in **CODE CELL 7** of the Jupyter notebook P5.jpynb.
Function `get_training_data`.

The combined features have been normalized by `StandardScaler() from the package sklearn`.

The following example shows the result:
![normalize](https://github.com/gada1982/CarND-Vehicle-Detection/blob/master/info_for_readme/histogramm.png)








