import dlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import os
import time

#from pyimagesearch.helpers import convert_and_trim_bb

#Euclidean Distance
def distances(points):
    dist = []
    for i in range(points.shape[0]):
        for j in range(points.shape[0]):
            p1 = points[i,:]
            p2 = points[j,:]
            dist.append( math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) )
    return dist

def get_bounding_box(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) 
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y 
	return x, y, w, h

def shape_to_np(shape, num_coords, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((num_coords, 2), dtype=dtype)
 	# loop over the facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, num_coords):
    #for i in range(27,68):
		coords[i] = (shape.part(i).x, shape.part(i).y) 
	# return the list of (x, y)-coordinates
	return coords

def get_landmarks(images, labels, save_directory="", num_coords=5, to_save=False):
    
    print("Getting %d facial landmarks" % num_coords)
    print('Number of total images %d' % len(labels))
    landmarks = []
    new_labels = []
    img_ct = 0
    count = 0
    #estimate time process
    start_time = time.time()
    
    modelPath = './mmod_human_face_detector.dat'
    if num_coords == 5:
        predictor_path = 'shape_predictor_5_face_landmarks.dat'
    else:
        predictor_path = 'shape_predictor_68_face_landmarks.dat'
    
    #for HOG
    detector = dlib.get_frontal_face_detector()
    #FOR CNN
    #detector = dlib.cnn_face_detection_model_v1(modelPath)
    predictor = dlib.shape_predictor(predictor_path)
    
    for img, label in zip(images, labels):
        
        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        
        img_ct += 1
        
        detected_faces = detector(img, 1)
        
        #show image
        # win = dlib.image_window(img)
        # win.add_overlay(detected_faces) 
        # win.wait_until_closed()        
        
        for d in detected_faces:
            
            count +=1
            new_labels.append(label)
            
            #Get the landmarks/parts for the face in box d.
            #for HOG
            x, y, w, h  = get_bounding_box(d)
            points = shape_to_np(
                    predictor(img, d), 
                    num_coords)
            
            #for cnn
            # x, y, w, h  = get_bounding_box(d.rect) 
            # points = shape_to_np(predictor(img, d.rect), num_coords)
                        
            dist = distances(points)                   
            landmarks.append(dist)    
            
            if to_save:
                for (x_, y_) in points:
                    cv2.circle(img, 
                           (x_, y_), 
                           1, 
                           (255, 0, 0), 
                           -1)
                
                plt.figure()
                plt.imshow(img)
                if not os.path.isdir(save_directory):
                    os.mkdir(save_directory)
                
                plt.savefig(save_directory + label + '%d.png' % img_ct, dpi=500)
                plt.close()
                
            if img_ct % 50 == 0:
                print("%d images with facial landmarks completed." % img_ct)
    
    percentage = count * 100 / len(images)
    print('Image detected : %d / %d. (%.1f percent)' % (count, len(images), percentage))
    print("Time for get_landmarks: --- %.2f minutes ---" % ((time.time() - start_time) / 60))
    
    return np.array(landmarks), np.array(new_labels)
            
            
        