import glob
import os
import myP1Lib
import matplotlib.image as mpimg

image_paths = glob.glob('test_images/*.jpg')
outdir = 'annotated_test_images'
if os.path.isdir(outdir) == False :
    raise RuntimeError('output dir ' + outdir + ' does not exist')

for image_path in image_paths :
    print('processing ' + image_path) 
    image = mpimg.imread(image_path)
    annotated_image = myP1Lib.process_image(image)
    image_name = os.path.basename(image_path)
    out_image_path = outdir + '/' + image_name
    mpimg.imsave(out_image_path, annotated_image)
