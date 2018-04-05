# **Finding Lane Lines on the Road** 

This is a writeup for project 1 finding and annotating lane lines on the road

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Making a pipeline that finds lane lines on the road
* Reflect on your work in a written report

[//]: # (Image References)

[input_image]: ./test_images/solidWhiteRight.jpg "Example input image"
[image1]: ./pipeline_stages/grayscale.jpg "Grayscale"
[image2]: ./pipeline_stages/blur_grayscale.jpg "Blurred_Grayscale"
[image3]: ./pipeline_stages/edges.jpg "Canny edges"
[image4]: ./pipeline_stages/masked_edges.jpg "Masked edge image"
[annotated_image]: ./annotated_test_images/solidWhiteRight.jpg "annotated output image"


---

The image pipeline was developed and applied to images as well as videos.

### Reflection

### 1. Description of the image pipeline.

My pipeline consisted of following steps. The output of each stage is also shown below for one example input image 

Example input image :
![alt text][input_image]

1. Generate a grayscale version of the input image, using helper function
 ... ![alt text][image1]
2. Then a Gaussian blur was applied with kernel size 5 to eliminate some noise
 ... ![alt text][image2]
3. Then Canny edge detection was applied with high threshold of 170, and low threshold of 60
 ... ![alt text][image3]
4. Then the resulting image is masked using a quadrilateral with vertices (0, 540), (450, 320), (500, 320) and (900, 540)
 ... ![alt text][image4]
5. Then hough transform is applied using the helper function. The parameters are tuned to get a good output as seen visually.
6. The output of hough transform is an image with lines or line segments drawn, which was then superimposed on the input image and saved as output. 
 ... ![alt text][annotated_image]

Initially, I used the draw_lines() method provided to draw the detected segmented lines. Then I replaced the draw_lines() function by draw_full_lines() with following changes: 
* draw_full_lines() function first find slopes of the lines. Lines with +ve slopes were classfied as belonging to right lane line while lines with -ve slopes are classfied as left lane lines. I filtered out those lines for which either slope was undefined (x1 == x2) or slope was < 20 degrees or more than 160 degrees. This is to prevent picking up some small lines which may have a +ve or -ve slope and get identified as part of right or left lane lines 
* then draw_full_lines() uses another function find_full_lines() which takes all left line segments and fit them into a linear model using np.polyfit and returns a line stretching from the bottom of the image to top of the masked region. Same was done for right line segments.
* Finally the extraploted left and right line were drawn on a blank image. 

### 2. Potential shortcomings of the current pipeline

I find the current pipeline is not very robust in presence of a) curved lines, b) small line segments, which may be part of left lines, but with negative slope and get mis-classified as part of line, modifying the final right line annotation significantly. Because of Canny and Hough transform parameter tuning, some thin lane line segments would not be picked up. 

### 3. Suggest possible improvements to your pipeline

In order to improve the pipeline, we need to tune Canny such that small line segments are picked up. But then, after Hough transform, the lines must be filtered out based on their slope. It may be possible to first classify the line segments as right and left, and then do a k-means clustering with k=1 for each bucket based on the slope of the line. The outliers should be filtered out as noise.

