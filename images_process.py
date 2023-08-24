# Load imports
import os 
import cv2
import time


def improve_contrast_image_using_clahe(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_planes = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv_planes[2] = clahe.apply(hsv_planes[2])
    hsv = cv2.merge(hsv_planes)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def resize_img(image, width, height):    
    dim = (width, height) 
    # resize image
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

'''
function get_images() definition
Parameter, image_directory, is the directory 
holding the images
'''

def get_images(image_directory):
    X = []
    y = []
    extensions = ('jpg','png','gif')
    #estimate time process
    start_time = time.time()
    
    #custom size of the image
    width = 150
    height = 150
    print('Image Size = %d x %d' %(width, height) )
    subfolders = os.listdir(image_directory)
    for subfolder in subfolders:
        print("Loading images in %s" % subfolder)
        if os.path.isdir(os.path.join(image_directory, subfolder)): # only load directories
            subfolder_files = os.listdir(
                    os.path.join(image_directory, subfolder)
                    )
            for file in subfolder_files:
                if file.endswith(extensions): # grab images only
                    # read the image using openCV                    
                    img = cv2.imread(
                            os.path.join(image_directory, subfolder, file)
                            )
                    # resize the image
                    img = resize_img(img, width , height)
                    
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    #cv2.imshow('Photo1', gray)
                    # #if the image is dark
                    if file.startswith("L_D"):                         
                        img = cv2.equalizeHist(gray)
                        
                    else:
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        img = clahe.apply(gray)
                                        
                    #cv2.imshow('Photo', img)
                    #cv2.waitKey()
                    X.append(img)
                    # add the image's label to a list y
                    #ID = int(subfolder_files.split('\\')[1])
                    y.append(subfolder)
                    
    print("Time for get_image: --- %.2f minutes ---" % ((time.time() - start_time) / 60))
    print("All images are loaded")     
    # return the images and their labels      
    return X, y

     
            