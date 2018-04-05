import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import glob
import os

mask_height = 320

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_left_right_lines(img, left_lines, right_lines, color=[255, 0, 0], thickness=5):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in left_lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    for line in right_lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), [0, 255, 0], thickness)

def line_slope(line):
    if len(line) != 1 :
        raise RuntimeError('must contain just one line here ' + str(line));
    line = line[0]
    (x1, y1, x2, y2) = line    
    if x1 == x2 :
        return None
    return (float(y2 - y1)/(x2 - x1))

def find_full_line(lines, imshape, isLeft) :
    x = []
    y = []
    #print('find_full_lines')
    if len(lines) == 0 :
        return np.array([]) 
    for line in lines :
        #print(line)
        for x1, y1, x2, y2 in line :
            x.append(x1)
            y.append(y1) 
            x.append(x2)
            y.append(y2)

    m, b = np.polyfit(x, y, 1)

    full_line_y1 = imshape[0] # image bottom 
    full_line_x1 = int(float(full_line_y1 - b)/m)
    full_line_y2 = mask_height 
    full_line_x2 = int(float(full_line_y2 - b)/m)
    return np.array([[[full_line_x1, full_line_y1, full_line_x2, full_line_y2]]]) 

def draw_full_lines(img, lines):
    right_lines = []
    left_lines = []
    for line in lines:
        slope = line_slope(line)
        if slope == None :
            #print('infinite slope for line ' + str(line))
            pass  
        elif slope == 0 :
            pass
            #print('0  slope for line ' + str(line))
        elif ((slope < 0.35) and (slope > -0.35)) :
            pass 
            #print ('slope too small ' + str(slope))  
        elif slope > 0 : # slope > 0 indicates right lines, because (0,0) is top left
            #print('right slope=' + str(slope))
            x1, y1, x2, y2 = line[0]
            right_lines.append([[x1, y1, x2, y2]])
        else :
            #print('left slope=' + str(slope))
            x1, y1, x2, y2 = line[0]
            left_lines.append([[x1, y1, x2, y2]])

    left_lines = np.array(left_lines)
    right_lines = np.array(right_lines)

    # find full lines
    full_left_lines = find_full_line(left_lines, img.shape, isLeft=True)
    #print ("full left lines : " + str(full_left_lines)) 
    full_right_lines = find_full_line(right_lines, img.shape, isLeft=False)
    #print ("full right lines : " + str(full_right_lines)) 
 
    draw_left_right_lines(img, full_left_lines, full_right_lines)
            
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_full_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)



def process_image(image) :

    # grayscale image
    gray_image = grayscale(image)
    mpimg.imsave('pipeline_stages/grayscale.jpg', gray_image, cmap='gray')
    
    # Guassian blur
    kernel_size = 5
    blur_gray_image = gaussian_blur(gray_image, kernel_size)
    mpimg.imsave('pipeline_stages/blur_grayscale.jpg', blur_gray_image, cmap='gray')
    
    # Canny edge detection
    low_threshold = 60 
    high_threshold = 170
    edge_image = canny(blur_gray_image, low_threshold, high_threshold)
    mpimg.imsave('pipeline_stages/edges.jpg', edge_image, cmap='gray')
    
    # mask image
    # vertices is an array of polygon, each poly consist of an array of vertices
    # each vertex consist of (x, y)
    imshape = edge_image.shape
    vertices = [[(0, imshape[0]), (450, mask_height), (500, mask_height), (900,imshape[0])]]
    vertices = np.array(vertices, dtype=np.int32)
    masked_edge_image = region_of_interest(edge_image, vertices)
    mpimg.imsave('pipeline_stages/masked_edges.jpg', masked_edge_image, cmap='gray')
    #return masked_edge_image
    #color_masked_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    #color_masked_img[:,:,0] = masked_edge_image
    
    # hough transform
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 10     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 40 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments
    hough_image = hough_lines(masked_edge_image, rho, theta, threshold, min_line_len, max_line_gap)
    annotated_image = weighted_img(hough_image, image)
    return annotated_image
