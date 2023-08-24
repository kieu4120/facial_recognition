''' Imports '''
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import performance
import images_process
import dectection_n_landmarks_process
import pandas

from sklearn.multiclass import OneVsRestClassifier as ORC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
''' Import classifier '''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#collect genuine scores and imposter score
def score_process(matching_scores, y_test):
    gen_scores = []
    imp_scores = []
    for i in range (len(y_test)):
        scores = matching_scores.loc[i]
        mask = scores.index.isin([y_test[i]])
        
        gen_scores.extend(scores[mask])
        imp_scores.extend(scores[~mask])
    
    return gen_scores, imp_scores

#evaluate the accuracy of the classifier
def classifier_accuracy(clf, X_test, y_test, method = ''):
    # make predictions
    yhat = clf.predict(X_test)
    # evaluate predictions
    accuracy = accuracy_score(y_test, yhat)
    print('Accuracy of %s: %.5f' % (method, accuracy))
    
    
def classifier_process(clf, count, method = ''):
    #split data to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    clf.fit(X_train, y_train)
    
    print('Number of trains:', len(y_train))
    print('Number of tests: ', len(y_test))
    
    #evaluate the accuracy of the classifier
    classifier_accuracy(clf, X_test, y_test, method)
    
    #get the matching score by predict the probability of X test
    matching_scores = clf.predict_proba(X_test)
    
    classes = clf.classes_

    matching_scores = pandas.DataFrame(matching_scores, columns = classes)
    
    gen_scores, imp_scores = score_process(matching_scores, y_test)
    
    count += 1
    
    return gen_scores, imp_scores, matching_scores, count, y_test


''''''''''''''''''''''''''''''''''''''''''''''''''''''''    

# NB, SVM, ANN
#directory.
image_directory = 'Project 1 Database'

#load images and enhance them
#load the data.
X, y = images_process.get_images(image_directory)


''' Get distances between face landmarks in the images '''
# get_landmarks(images, labels, save_directory="", num_coords=5, to_save=False)
X, y= dectection_n_landmarks_process.get_landmarks(X, y, "landmarks/", 68, False)


''''''''''''''''''''''''''''''''''''''''''''''''
print('System:')
number_of_classifier = 0
# create an instance of the classifier
clf = ORC(SVC(probability=True))
method = 'SVC'

#GET THE THINGS FROM THE CLASSIFIER PROCESS: 

#USING THE CLASSIFIER
gen_scores, imp_scores, matching_scores, number_of_classifier, y_test = classifier_process(clf, number_of_classifier, method)

clf2 = ORC(KNeighborsClassifier(3))
method2 = 'KNN(k = 3)'
gen_scores2, imp_scores2, matching_scores2, number_of_classifier, y_test = classifier_process(clf2, number_of_classifier, method2)

clf3 = ORC(KNeighborsClassifier(7))
method3 = 'KNN(k = 7)'
gen_scores3, imp_scores3, matching_scores3, number_of_classifier, y_test = classifier_process(clf3, number_of_classifier, method3)


#if there are 3 classifiers
if number_of_classifier == 3:
    #apply score_fusion to get the optimal option (get the average) - combining the score. 
    matching_scores_avg = (matching_scores + matching_scores2 + matching_scores3) / number_of_classifier
    
    #collect genuine and impostor scores of the optimal case - BY USING THE THRESHOLD SCORE TO TEST ON THE INPUT DATA. 
    gen_scores_avg, imp_scores_avg = score_process(matching_scores_avg, y_test)
    method_avg = 'Optimization System'
    
    #use classifiers for performance
    performance.performance(gen_scores, imp_scores, gen_scores2, imp_scores2, gen_scores3, imp_scores3, 
                            gen_scores_avg, imp_scores_avg ,method, method2, method3, method_avg, 500)


