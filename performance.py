import numpy as np
import matplotlib.pyplot as plt
import math
import os
from sklearn import metrics

############ functions #############################################################

def dprime(gen_scores, imp_scores):
    x = math.sqrt(2) * abs(np.mean(gen_scores) - np.mean(imp_scores)) # replace 1 with the numerator
    y = math.sqrt(pow(np.std(gen_scores), 2) + pow(np.std(imp_scores), 2)) # replace 1 with the denominator
    return x / y

def plot_scoreDist(gen_scores, imp_scores, plot_title, decision_threshold):
    fig = plt.figure()
    
    #draw threshold decision    
    plt.axvline(x = decision_threshold, ymin = 0, ymax = 0.5, linestyle = '--', label = 'Decision Threshold')      
    plt.text(decision_threshold + 0.05, 400, "Decision Threshold = %.3f" % decision_threshold)
    #genuine histogram
    plt.hist(gen_scores, color = 'green', lw = 2,
             histtype= 'step', hatch = '//', label = 'Genuine Scores')
    #impostor histogram
    plt.hist(imp_scores, color = 'red', lw = 2,
             histtype= 'step', hatch = '\\', label = 'Impostor Scores')
    plt.xlim([-0.05,1.05])
    plt.ylim([0, 1000])
    #find the best location for naming
    plt.legend(loc = 'best')
    #estimate d-prime
    dp = dprime(gen_scores, imp_scores)
    #add title of the graph
    plt.title(plot_title + '\nD-prime = %.2f' % dp)    
    plt.show()
    #save the graph
    fig.savefig('./charts/Score_dist_%s.png' % plot_title, dpi = 96)
    plt.close()
    return

def get_EER(far, frr, thresholds):
    eer = 0
    '''
        Use the FARs and FRRs to return the error
        in which they are approximately equal.
        
    '''
        
    dis = []
    for i in range(len(far)):
        dis.append(abs(far[i] - frr[i]))
    eer_index = np.argmin(dis)
    decision_threshold = thresholds[eer_index]
    eer = (far[eer_index] + frr[eer_index]) / 2
                  
    return eer, decision_threshold

#Detection Error Tradeoff 
def plot_det(far, frr, far2, frr2, far3, frr3, far_avg, 
             frr_avg, plot_title1, plot_title2, plot_title3, plot_title_avg, thresholds):
    title = 'Detection Error Tradeoff Curve (DET)'
    #calculate err for each case
    eer, decision_threshold = get_EER(far, frr, thresholds)
    eer2, decision_threshold2 = get_EER(far2, frr2, thresholds)   
    eer3, decision_threshold3 = get_EER(far3, frr3, thresholds)
    eer_avg, decision_threshold_avg = get_EER(far_avg, frr_avg, thresholds)
              
    fig = plt.figure()
    
    #draw the curve of each classifier
    plt.plot(far,frr, lw = 2, label = plot_title1)
    plt.plot(far2,frr2, lw = 2, label = plot_title2)
    plt.plot(far3,frr3, lw = 2, label = plot_title3)
    plt.plot(far_avg, frr_avg, lw = 2, label = plot_title_avg)
    #find the best location
    plt.legend(loc= 'best')
    
    eer_min = min(eer, eer2, eer3, eer_avg)
    #dot the best EER in the graph
    plt.scatter([eer_min], [eer_min], c = 'black' ,s = 90)
    plt.text(eer_min + 0.1, eer_min + 0.1, "Best EER = %.3f" % eer_min )
    
    plt.plot([0,1], [0,1], lw = 1, color = 'black')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    #label x and y axis
    plt.xlabel('FAR')
    plt.ylabel('FRR')
    #add the title for DET graph
    plt.title(title + '\nEER of %s = %.3f, %s = %.3f \n%s = %.3f, %s = %.3f' 
              % (plot_title1, eer, plot_title_avg, eer_avg, plot_title2, eer2, plot_title3, eer3))
    
    plt.show()
    #save the figure
    fig.savefig('./charts/DET_%s.png' % title, dpi = 96)
    plt.close()
    return decision_threshold, decision_threshold2, decision_threshold3, decision_threshold_avg

#Receiver Operating Characteristic
def plot_roc(far, tpr, far2, tpr2, far3, tpr3, far_avg, tpr_avg, 
             plot_title1, plot_title2, plot_title3, plot_title_avg):
    title = 'Receiver Operating Characteristic Curve (ROC)'
    fig = plt.figure()
    '''
        Refer back to lecture for ROC curve
    '''
    #Find Area Under Curve
    auc1 = metrics.auc(far, tpr)    
    auc2 = metrics.auc(far2, tpr2)
    auc3 = metrics.auc(far3, tpr3)
    auc_avg = metrics.auc(far_avg, tpr_avg)
    
    #Find maximum AUC
    auc = max(auc1, auc2, auc3, auc_avg)
    
    
    #draw the ROC curves
    plt.plot(far, tpr, lw = 2, label = plot_title1 + ': %.3f' % auc1)
    plt.plot(far2, tpr2, lw = 2, label = plot_title2 + ': %.3f' % auc2)
    plt.plot(far3, tpr3, lw = 2, label = plot_title3 + ': %.3f' % auc3)
    plt.plot(far_avg, tpr_avg, lw = 2, label = plot_title_avg + ': %.3f' % auc_avg)
    
    #find the best location
    plt.legend(loc= 'best')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05, 1.05])
    #label x and y axis
    plt.xlabel('FAR')
    plt.ylabel('TAR')
    
    #add the title
    plt.title(title + '\nBest Area Under Curve = %.3f' % auc)
    
    plt.show()
    #save figure
    fig.savefig('./charts/ROC_%s.png' % title, dpi = 96)
    plt.close()
    return

# Function to compute TPR, FAR, FRR
def compute_rates(gen_scores, imp_scores, thresholds):
    
    # use np.linspace to create n threshold values 
                 # between 0 and 1
    far = [] #False Positive Rate
    frr = [] #False Negative Rate
    tpr = [] #True Positive Rate
    
    
    for t in thresholds:
        '''
            Initialize tp, fp, tn, fn            
        '''
        tp, fp, tn, fn = 0,0,0,0
        
        for g_s in gen_scores:
            '''
                Count tp and fn
            '''
            if g_s >= t:
                tp += 1
            else:
                fn += 1
                
        for i_s in imp_scores:
            '''
                Count tn and fp
            '''
            if i_s >= t:
                fp += 1
            else:
                tn += 1
        #calculate the rate
        far.append(fp / (fp + tn)) #equation for far
        frr.append(fn / (fn + tp)) #equation for frr
        tpr.append(tp / (tp + fn)) #equation for tpr
    return far, frr, tpr

############ main code #############################################################

def performance(gen_scores, imp_scores, gen_scores2, imp_scores2, 
                gen_scores3, imp_scores3, gen_scores_avg, imp_scores_avg, 
                plot_title1, plot_title2, plot_title3, plot_title_avg, num_thresholds):   
    # start at 0 to 1, number of num = 500
    thresholds =  np.linspace(0.0, 1.0, num_thresholds)
    #check the directory exist or not
    if not os.path.exists('charts'):
        os.makedirs('charts')
    
    #find far, frr and tpr for each case of classifier
    far, frr, tpr = compute_rates(gen_scores, imp_scores, thresholds) #parameters
    far2, frr2, tpr2 = compute_rates(gen_scores2, imp_scores2, thresholds)
    far3, frr3, tpr3 = compute_rates(gen_scores3, imp_scores3, thresholds)
    far_avg, frr_avg, tpr_avg = compute_rates(gen_scores_avg, imp_scores_avg, thresholds)
    
    
    plot_roc(far, tpr, far2, tpr2, far3, tpr3, far_avg, tpr_avg,
             plot_title1, plot_title2, plot_title3, plot_title_avg) #parameters
    
    decision_threshold, decision_threshold2, decision_threshold3, decision_threshold_avg = plot_det(
        far, frr, far2, frr2, far3, frr3, far_avg, frr_avg, 
        plot_title1, plot_title2, plot_title3, plot_title_avg, thresholds) #parameters
    
    plot_scoreDist(gen_scores, imp_scores, plot_title1, decision_threshold) #parameters
    plot_scoreDist(gen_scores2, imp_scores2, plot_title2, decision_threshold2) #parameters
    plot_scoreDist(gen_scores3, imp_scores3, plot_title3, decision_threshold3) #parameters
    plot_scoreDist(gen_scores_avg, imp_scores_avg, plot_title_avg, decision_threshold_avg) #parameters
    

