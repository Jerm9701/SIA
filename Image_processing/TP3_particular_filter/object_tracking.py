import numpy as np
import random
import cv2
import math
import os
from time import sleep

"""Histogram calculation"""
def initialize_href(img,init_center,size_rectangle=[20,20],Nb_bins = 25):
    sub_image = img[init_center[0]-size_rectangle[0]:init_center[0]+size_rectangle[0],
    init_center[1]-size_rectangle[1]:init_center[1]+size_rectangle[1]]
    href, bin_edges = np.histogram(sub_image, bins=Nb_bins, range=(0, 255))
    nb_pixels = size_rectangle[0] * size_rectangle[1] * 4 * 3
    href = href / nb_pixels
    return href

def initialize_href_by_channel (img,init_center,size_rectangle = [20,20],Nb_bins=25):
    sub_image = img[init_center[0]-size_rectangle[0]:init_center[0]+size_rectangle[0],init_center[1]-size_rectangle[1]:init_center[1]+size_rectangle[1]]
    href_1, bin_edges = np.histogram(sub_image[:,:,0], bins=Nb_bins, range=(0, 255))
    href_2, bin_edges = np.histogram(sub_image[:,:,1], bins=Nb_bins, range=(0, 255))
    href_3, bin_edges = np.histogram(sub_image[:,:,1], bins=Nb_bins, range=(0, 255))
    href = (href_1+href_2+href_3)/(size_rectangle[0] * size_rectangle[1] * 4 * 3)
    return href

def initialize_href_angle(img,init_center,size_rectangle=[20,20],angle=0,Nb_bins = 25):
    sub_image = img[init_center[0]-size_rectangle[0]:init_center[0]+size_rectangle[0],init_center[1]-size_rectangle[1]:init_center[1]+size_rectangle[1]]
    rot_mat = cv2.getRotationMatrix2D(init_center,angle,1)
    sub_image = cv2.rotate(img,rot_mat)
    href, bin_edges = np.histogram(sub_image, bins=Nb_bins, range=(0, 255))
    nb_pixels = size_rectangle[0] * size_rectangle[1] * 4 * 3
    href = href / nb_pixels
    return href

def initialize_particles(N, center):
    weights = np.ones(N)/N
    positions = []
    for k in range(N):
        positions.append([center[0] ,center[1]])
    return positions, weights

def initialize_angled_particles(N,center):
    weights = np.ones(N)/N
    positions = []
    angles = np.zeros(N)
    for k in range(N):
        positions.append([center[0] ,center[1]])
    return positions, weights,angles

"""Transition function"""
#Adds a random noise to the previous positions to try to predict the current location
def prediction_step(position,deviation=15 ) : 
    position = [int(random.gauss(position[0],deviation)),int(random.gauss(position[1],deviation))]
    return position

def prediction_step_angled(position,angle,deviation=15 ) :
    position = [int(random.gauss(position[0],deviation)),int(random.gauss(position[1],deviation))]
    angle = np.arange(deviation)*9
    return position,angle

"""Likelihood function"""
#Corrects the prediction using the observation of the object
def distance(href,hpred) : 
    dist = 0
    hsum=0
    for i in range(len(href)) :
        hsum += (href[i]*hpred[i])**0.5
    dist = (1-hsum)**0.5
    return dist

def likelihood(dist,lmbd=5):
    lklh = np.exp(-lmbd*dist**2)
    return lklh
#lmbd not too high to keep enough particles 

def run_ini(img, init_center =[320,240], size_rectangle = [20,20],N=20):
    href = initialize_href(img,init_center,size_rectangle)
    random_positions,weights = initialize_particles(N,init_center)
    return href,weights,init_center,random_positions

"""Resampling function"""
#Uses systematic resampling when the weights values start to dip : Replaces the lower weight values by values greater than the threshold
def resampling(new_weights,predicted_positions,threshold) :
    resampled_weights = new_weights.copy()
    resampled_positions = predicted_positions.copy()
    stable_weights = np.where(new_weights>=threshold)[0]
    for resampled_indices in np.where(new_weights<threshold)[0] :
        random_weight_indice = random.choices(stable_weights,new_weights[stable_weights],k=1)[0]
        resampled_weights[resampled_indices]=new_weights[random_weight_indice]
        resampled_positions[resampled_indices]=predicted_positions[random_weight_indice]
    resampled_weights = resampled_weights/sum(resampled_weights)
    return resampled_weights,resampled_positions

def display_particles(img,predicted_positions,estimated_new_center,new_weights,size_rectangle = [20,20]) :
    img = cv2.flip(img,1)
    for pos in predicted_positions :
        cv2.rectangle(img, (pos[1] - size_rectangle[1], pos[0] - size_rectangle[0]),
                          (pos[1] + size_rectangle[1], pos[0] + size_rectangle[0]), (0, 255, 0), 1)
    max_weight = np.argmax(new_weights)
    cv2.rectangle(img, (estimated_new_center[1] - size_rectangle[1], estimated_new_center[0] - size_rectangle[0]),
                          (estimated_new_center[1] + size_rectangle[1], estimated_new_center[0] + size_rectangle[0]), (0, 255, 255), 2)
    cv2.rectangle(img, (estimated_new_center[1]-4, estimated_new_center[0]+4),
                          (estimated_new_center[1]-4, estimated_new_center[0]+4), (0, 0, 255), 2)

    cv2.imshow('image',img)
    cv2.waitKey(300)
    cv2.destroyAllWindows()

def display_angled_particles(img,predicted_positions,estimated_new_center,new_weights,estimated_angle,size_rectangle = [20,20]) :
    img = cv2.flip(img,1)
    for pos in predicted_positions :
        cv2.rectangle(img, (pos[1] - size_rectangle[1], pos[0] - size_rectangle[0]),
                          (pos[1] + size_rectangle[1], pos[0] + size_rectangle[0]), (0, 255, 0), 1)
    cv2.rectangle(img, (estimated_new_center[1] - size_rectangle[1], estimated_new_center[0] - size_rectangle[0]),
                          (estimated_new_center[1] + size_rectangle[1], estimated_new_center[0] + size_rectangle[0]), (0, 255, 255), 2)
    cv2.rectangle(img, (estimated_new_center[1]-4, estimated_new_center[0]+4),
                          (estimated_new_center[1]-4, estimated_new_center[0]+4), (0, 0, 255), 2)

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""Calculates the new center using the weighted average of predicted points"""
def main_loop(img,new_weights,predicted_positions,estimated_new_center,size_rectangle,href,N=100):
    for position in range (len(predicted_positions)) :
        predicted_positions[position]=prediction_step(estimated_new_center)
    hist_dist = []
    estimated_likelihood = []
    indice = 0
    for position in predicted_positions :
        predicted_hist =initialize_href(img,position,size_rectangle)
        hist_dist.append(distance(href,predicted_hist))
        estimated_likelihood.append(likelihood(hist_dist[indice]))
        new_weights[indice] = new_weights[indice]*estimated_likelihood[indice]
        indice+=1
    new_weights = new_weights/sum(new_weights)
    estimated_new_center = [0,0]
    for index in range(len(predicted_positions)) :
        estimated_new_center[0] +=new_weights[index]*predicted_positions[index][0]
        estimated_new_center[1] +=new_weights[index]*predicted_positions[index][1]
    estimated_new_center[0] = int(estimated_new_center[0])
    estimated_new_center[1] = int(estimated_new_center[1])
    if len(np.where(new_weights<0.05)[0])>0:
        new_weights,predicted_positions=resampling(new_weights,predicted_positions,threshold=0.05)
    display_particles(img,predicted_positions,estimated_new_center,new_weights,size_rectangle)
    return href,new_weights,estimated_new_center,predicted_positions

"""Using rotation methods"""
def main_loop_rotate(img,new_weights,predicted_positions,predicted_angles,estimated_new_center,size_rectangle):
    for position in range (len(predicted_positions)) :
        predicted_positions[position]=prediction_step_angled(estimated_new_center,angle=predicted_angles[position])
    hist_dist = []
    estimated_likelihood = []
    indice = 0
    for position in predicted_positions :
        predicted_hist =initialize_href_angle(img,position,size_rectangle,predicted_angles[position])
        hist_dist.append(distance(href,predicted_hist))
        estimated_likelihood.append(likelihood(hist_dist[indice]))
        new_weights[indice] = new_weights[indice]*estimated_likelihood[indice]
        indice+=1
    new_weights = new_weights/sum(new_weights)
    estimated_new_center = [0,0]
    estimated_new_angle = 0
    for index in range(len(predicted_positions)) :
        estimated_new_center[0] +=new_weights[index]*predicted_positions[index][0]
        estimated_new_center[1] +=new_weights[index]*predicted_positions[index][1]
        estimated_new_angle += new_weights[index]*predicted_angles[index]
    estimated_new_center[0] = int(estimated_new_center[0])
    estimated_new_center[1] = int(estimated_new_center[1])
    estimated_new_angle = np.mod(estimated_new_angle,np.pi)
    
    if len(np.where(new_weights<0.05)[0])>0:
        new_weights,predicted_positions=resampling(new_weights,predicted_positions,threshold=0.05)
    display_particles(img,predicted_positions,estimated_new_center,new_weights,size_rectangle)
    return href,new_weights,estimated_new_center,predicted_positions

def draw_angled_rec(x0, y0, width, height, angle, img):

    _angle = angle * math.pi / 180.0
    b = math.cos(_angle) * 0.5
    a = math.sin(_angle) * 0.5
    pt0 = (int(x0 - a * height - b * width),
           int(y0 + b * height - a * width))
    pt1 = (int(x0 + a * height - b * width),
           int(y0 - b * height - a * width))
    pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
    pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

    cv2.line(img, pt0, pt1, (255, 255, 255), 3)
    cv2.line(img, pt1, pt2, (255, 255, 255), 3)
    cv2.line(img, pt2, pt3, (255, 255, 255), 3)
    cv2.line(img, pt3, pt0, (255, 255, 255), 3)


if __name__ == '__main__':
    test_img_path = 'data/video sequences/output_images/escrime-4-3-cluster'
    test_img_list = os.listdir(test_img_path)   
    test_img_1 = cv2.imread(os.path.join(test_img_path,test_img_list[0]))
    size_rectangle=[20,20]
    init_center=[240,320]
    true_cernter = [np.linspace(240,200,50).astype(np.int16),np.linspace(320,374,50).astype(np.int16)]
    N=10
    href,new_weights,estimated_new_center,predicted_positions=run_ini(test_img_1,init_center,size_rectangle,N)
    display_particles(test_img_1,predicted_positions,estimated_new_center,new_weights,size_rectangle)
    count = 0
    for images in test_img_list[0:50]:
        img = cv2.imread(os.path.join(test_img_path,images))
        print(images)
        href = initialize_href_by_channel(img,[true_cernter[0][count],true_cernter[1][count]],size_rectangle)
        href,new_weights,estimated_new_center,predicted_positions = main_loop(img,new_weights,predicted_positions,estimated_new_center,size_rectangle=size_rectangle,N=10,href=href)
        print(estimated_new_center)
        print(new_weights)
        count+=1

#img100 : 200,374
#Alternative to histogram likelihood function : compare the nb of pixels of the particle that match with the target object