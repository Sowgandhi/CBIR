import cv2
import numpy as np
import scipy
from scipy.misc import imread
import _pickle as pickle
import random
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import extract

class Matcher(object):

    def __init__(self, pickled_db_path="features_msrc.pickle"):
        with open(pickled_db_path, 'rb') as fp:
            self.data = pickle.load(fp)
        self.names = []
        self.matrix = []
        for k, v in self.data.items():
            self.names.append(k)
            self.matrix.append(v)
        self.matrix = np.array(self.matrix)
        self.names = np.array(self.names)

    def cos_cdist(self, vector):
        # getting cosine distance between search image and images database
        v = vector.reshape(1, -1)
        return scipy.spatial.distance.cdist(self.matrix, v, 'Euclidean').reshape(-1)

    def match(self, image_path, topn=5):
        features = extract.extract_features(image_path)
        print("start")
        print(len(features))
        print("End")
        print(len(self.names)  )
        print(len(self.matrix))
        img_distances = self.cos_cdist(features)
        #for img in img_distances:
        #    img = img/img_distances[len(img_distances - 2)]
        nearest_ids = np.argsort(img_distances).tolist()
        nearest_img_paths = self.names[nearest_ids].tolist()
        return nearest_img_paths, img_distances[nearest_ids].tolist(), nearest_ids


def show_img(path):
    image_query =  mpimg.imread(path['q'])
    #image_query = cv2.resize(image_query, (0, 0), None, 0.5, 0.5)

    image_result =  mpimg.imread(path['r'])
    #image_result = cv2.resize(image_result, (0, 0), None, 0.5, 0.5)


    cv2.imshow('query image', image_query)
    cv2.imshow('result images', image_result)
    cv2.waitKey()

def accuracy(img_dis, indexs):
    relevant_imgs = []
    i = 0
    for  val in img_dis:
        #########################################
        ##### (index, distance) ################
        relevant_imgs.append((indexs[i], val))
        i = i + 1
    relevant_imgs.sort()

    #########################################
    ########## (PRICISION, ACCURACY) /////////
    ######## precision = (1:num_relevant_images) ./ locations_sorted;
    ######## recall = (1:num_relevant_images) / num_relevant_images;
    precision_and_recall = []
    print(len(relevant_imgs))
    count = 0
    print("(precision                 ,           recall)")
    print("------------------------------------------------")
    precision1 = 0
    recall1 = 0
    for rel in relevant_imgs:
        count = count + 1
        precision = count/(rel[0] + 1)
        precision1= precision1+precision
        recall = count/len(relevant_imgs)
        recall1 = recall1 + recall
        print(f"{precision} , {recall}")
        precision_and_recall.append((precision, recall))
    print(precision1/len(relevant_imgs))
    print(recall1/len(relevant_imgs))
    print("-------------------------------------------------")

    return precision_and_recall


def run():
    images_path = './MSRC_v2/'
    images_path_training = './trai_msrc/'
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    # getting 1 random images
    sample = random.sample(files, 1)


    #extract.batch_extractor(images_path_training)

    ma = Matcher('features_msrc.pickle')


    print('Query image and result')

    # imgSam = '/home/ermicho/projects/python/imageExt/images/car-1.jpg'
    #print(sample[0])
    names, match, nearest_ids = ma.match(sample[0], topn=200)
    new_List = [i/match[-1] for i in match]
    #print(new_List)
    print(new_List)


   ############################################################
   ########### filter relevant matchs thrashold = 0.4 #########
    thrashold = 0.8
    relevant_match = list(filter(lambda x: x < thrashold, new_List))
    #print(relevant_match)
    l = len(relevant_match)
    relevant_ids = nearest_ids[: l]
    print(len(relevant_match))
    print(len(relevant_ids))
    #####  ##################################
    ######## calculate accuracy #####

    accuracy(relevant_match, relevant_ids)
    img = os.path.join(images_path_training, names[0])
    #print(img)
    res = {'q' : sample[0], 'r' : img}

    show_img(res)

    # for s in sample:
    #     print('Query image ==========================================')
    #     print(s)
    #     # show_img(s)
    #     names, match = ma.match(s, topn=3)
    #     print('Result  ========================================')
    #     img = os.path.join(images_path, names[0])
    #     print(img)

        # for i in range(1):
        #     # we got cosine distance, less cosine distance between vectors
        #     # more they similar, thus we subtruct it from 1 to get match value
        #     print('Match %s' % (1 - match[i]))
        #     # show_img(os.path.join(images_path, names[i]))
        #     img = os.path.join(images_path, names[i])
        #     print(img)


run()