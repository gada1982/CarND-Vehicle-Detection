# Introduction

This project is done as a part of the **Nanodegree - Self-Driving Car Engineer** provided by Udacity. The scope of this project is finding and marking vehicles on a given, recorded track. The recorded raw video and some used sample codes are provided by Udacity.

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
3. Classifier
4. Sliding Window Search
5. Heat Map and Combined Box
6. Pipeline for Single Images
7. Pipeline for Video Stream
8. Conclusion

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
Out of the extracted features for cars and non-cars training and test data for the classifier has been generated. The code for this functionality can be found in **CODE CELL 7** of the Jupyter notebook P5.jpynb.
Function `get_training_data`.

The combined features have been normalized with `StandardScaler()` from the package `sklearn`.

The following example shows the result:
![normalize](https://github.com/gada1982/CarND-Vehicle-Detection/blob/master/info_for_readme/histogramm.png)

The whole data was split up into training and test data by using `train_test_split` from the package `sklearn`. 20% of the data is used for testing, and never user for training.

### 3.6 Training of the Classifier

A Support Vector Machines (SVM) is used for classification. `LinearSVC` from the package `sklearn` has been taken.
The code for this functionality can be found in **CODE CELL 8** of the Jupyter notebook P5.jpynb.

The final **accuracy** on the test data is **0.9896**.

### 3.7 Checkpoint

The whole classifier including its parameter is stored as a pickle-file to have a checkpoint. This helps to speed up the development process because the classifier can be taken as it is without training again and again.

The code for this functionality can be found in **CODE CELL 9** and **CODE CELL 10** of the Jupyter notebook P5.jpynb.
Functions `save_as_pickle` and `read_pickle`.

### 3.8 Validation of the classifier

The classifier was validate by a small set of images (which have not been used for training and testing at all). Eight car images and eight non-car-images have been choosen, the features have been extracted, predections have been made and the images have been labeled. One stands for car, zero for non-car classification.

The code for this functionality can be found in **CODE CELL 11** of the Jupyter notebook P5.jpynb.
Functions `extract_features_single`, `get_prediction_of_single_image` and `show_labeled_images_as_table`.

The following table shows the results:
![prediction_images](https://github.com/gada1982/CarND-Vehicle-Detection/blob/master/info_for_readme/predictions.jpg)

# 4. Sliding Window Search

A region of interest was defined where cars could appear. Sub-regions (windows) of the image have been taken and classified. If the single window has been classified as one (car found) a box has been drawn on the original Image. Different window sizes (64x64, 96x96, 140x140, 210x210 and 220x220) have been processed to make the detection-pipeline more robust.

The code for this functionality can be found in **CODE CELL 13 to 16** of the Jupyter notebook P5.jpynb.
Functions `return_parameter`, `slide_window_single`, `draw_boxes`, `slide_window_all`, `single_img_features`, `search_windows`.

Function `slide_window_single` defines sub-regions (windows) of the whole image in defined single sizes:
Sub-regions of different sizes have been added (overlapping) the respect different car sizes in different distances.
Sub-regions of the same size are overlapping (horizontally) by 80% to get multiple detections. The sub-regions are drawn with function `draw_boxes`.

The following examples show the overlapping sub-regions:
![boxes](https://github.com/gada1982/CarND-Vehicle-Detection/blob/master/info_for_readme/boxes.jpg)

Function `slide_window_all` does the work of multiple calls of function `slide_window_single` (with differnt sizes) in one step. The sub-regions are drawn with function `draw_boxes`.

Function `search_windows` takes the list of sub-regions (provided by function `slide_window_all`), does predictions on the single sub-regions with function `single_img_features`) and marks all sub-regions where a car was found as hot-windows. The hot sub-regions are drawn with function `draw_boxes`.

The following example show the result:
![overlapping](https://github.com/gada1982/CarND-Vehicle-Detection/blob/master/info_for_readme/overlapping.jpg)

# 5. Heat Map and Combined Box

To handle overlapping, multiple detections of the same car a heat map was introduced. Each single detection adds five 'points' to the heatmap of the pixels on which the detection occours. Out of the heatmap a combined box for each car is generated.

The code for this functionality can be found in **CODE CELL 17** and **CODE CELL 18** of the Jupyter notebook P5.jpynb.
Functions `add_heat`, `apply_threshold` and `draw_labeled_bboxes`.

The following image show the heatmap of the image above:
![heat_map](https://github.com/gada1982/CarND-Vehicle-Detection/blob/master/info_for_readme/heatmap.jpg)

With function `apply_threshold` all pixels are set to zero, which do not reach the minimum amount of 'points'.
This helps to prevent false positives. With function `draw_labeled_bboxes` a combined box is drawn around each single detected car.

The following image show the combined boxes for the detections on the image above:
![combined_box](https://github.com/gada1982/CarND-Vehicle-Detection/blob/master/info_for_readme/combined_box.jpg)

# 6. Pipeline for Single Images

Out of all this function a single pipeline was generated, which takes a single image and produces an images with drawn combined boxes around each detected car.

The code for this functionality can be found in **CODE CELL 19** of the Jupyter notebook P5.jpynb.
Function `generate_processed_image`.

This looks like the following example:
The following image show the combined boxes for the detections on the image above:
![combined_box](https://github.com/gada1982/CarND-Vehicle-Detection/blob/master/info_for_readme/combined_box.jpg)

# 7. Pipeline for Video Stream

The same pipeline as above is used for processing the video stream. Each framed is fed into the pipeline. To smoothen the output a multi-frame heatmap was introduced in the pipeline. The last ten frames are combined in a weighted heatmap. The older the frame is, the less weight it gets. This helps to prevent 'jumping' detection and false positives.

The code for this functionality can be found in **CODE CELL 20** and **CODE CELL 20** of the Jupyter notebook P5.jpynb.
The pipeline is applied to the 'test_video' and the 'project_video'.

The following video show the final project video:
[Project Video](https://youtu.be/IGq4IbWYJzc)

# 8. Conclusio





