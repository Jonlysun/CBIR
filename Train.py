# coding: utf-8
import cv2
import numpy as np
from PIL import Image
import argparse
import os
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


class Train():
    def __init__(self, dataset_path, num_words, method):
        self.dataset_path = dataset_path
        self.num_words = num_words
        self.method = method
        if self.method == 'sift' or self.method == 'test':
            self.feat_extract = cv2.xfeatures2d.SIFT_create()
            self.feature_num = 128
        elif self.method == 'surf':
            self.feat_extract = cv2.xfeatures2d.SURF_create()
            self.feature_num = 64
        elif self.method == 'orb':
            self.feat_extract = cv2.ORB_create(1000)
            self.feature_num = 32
        if not os.path.exists(self.method):
            os.makedirs(self.method)

    def get_img_path(self, training_path):
        print("dataset : {} is loading ......".format(training_path))
        classes = os.listdir(training_path)
        image_number = 0
        img_paths = []
        for each in classes:
            dirs = os.path.join(training_path, each)
            training_names = os.listdir(dirs)
            image_number += len(training_names)
            for name in training_names:
                img_path = os.path.join(dirs, name)
                img_paths.append(img_path)
        print('Image number: {}'.format(image_number))
        print('Classes number: {}'.format(len(classes)))
        return img_paths

    def getClusterCentures(self, img_paths, dataset_matrix, num_words):
        des_list = []  # 特征描述
        des_matrix = np.zeros((1, self.feature_num))
        # sift_det = cv2.xfeatures2d.SIFT_create()
        # sift_det = cv2.SIFT_create()
        count = 1
        print(f'{self.method} features extracting ......')

        # img_paths = img_paths[:20]

        if img_paths != None:
            for path in img_paths:
                img = Image.open(path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = np.array(img)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                kp, des = self.feat_extract.detectAndCompute(gray, None)
                if des != []:
                    des_matrix = np.row_stack((des_matrix, des))
                des_list.append(des)

                count += 1
                if count % 50 == 0:
                    print('\t{} has finished'.format(count))
        elif dataset_matrix != None:
            for gray in range(dataset_matrix.shape[0]):
                kp, des = self.feat_extract.detectAndCompute(gray, None)
                if des != []:
                    des_matrix = np.row_stack((des_matrix, des))
                des_list.append(des)
        else:
            raise ValueError('输入不合法')

        des_matrix = des_matrix[1:, :]   # the des matrix of sift

        # 计算聚类中心  构造视觉单词词典
        print('Calculate Kmeans center: ......')
        kmeans = KMeans(n_clusters=num_words, random_state=33)
        kmeans.fit(des_matrix)
        centres = kmeans.cluster_centers_  # 视觉聚类中心

        return centres, des_list

    def VLAD_des2feature(self, des, num_words, centures):
        img_features = np.zeros((num_words, self.feature_num), 'float32')
        for i in range(des.shape[0]):
            # select the nearest center
            feature_k_rows = np.ones((num_words, self.feature_num), 'float32')
            feature = des[i]
            feature_k_rows = feature_k_rows*feature
            feature_k_rows = np.sum((feature_k_rows-centures)**2, 1)
            index = np.argmax(feature_k_rows)
            nearest_center = centures[index]
            # caculate the residual
            residual = feature - nearest_center
            img_features[index] += residual

        norm = np.linalg.norm(img_features)
        img_features = img_features / norm

        # PCA TODO
        img_features = img_features.flatten()
        return img_features

    def des2feature(self, des, num_words, centures):
        img_feature_vec = np.zeros((1, num_words), 'float32')
        for i in range(des.shape[0]):
            feature_k_rows = np.ones((num_words, self.feature_num), 'float32')
            feature = des[i]
            feature_k_rows = feature_k_rows*feature
            feature_k_rows = np.sum((feature_k_rows-centures)**2, 1)
            index = np.argmax(feature_k_rows)
            img_feature_vec[0][index] += 1
        return img_feature_vec

    def get_all_features(self, des_list, num_words, centres):
        print(f'{self.method} feature encoding ......')
        allvec = np.zeros((len(des_list), num_words), 'float32')
        for i in range(len(des_list)):
            if des_list[i] != []:
                allvec[i] = self.des2feature(
                    centures=centres, des=des_list[i], num_words=num_words)

            if i % 50 == 0:
                print('\t{} encode has finished'.format(i))
        return allvec

    def getNearestImg(self, feature, dataset, num_close):
        features = np.ones((dataset.shape[0], len(feature)), 'float32')
        features = features*feature
        dist = np.sum((features-dataset)**2, 1)
        dist_index = np.argsort(dist)
        return dist_index[:num_close]

    def train(self):
        img_paths = self.get_img_path(self.dataset_path)
        np.save(os.path.join(self.method, 'image_paths.npy'), np.array(img_paths))

        centres, des_list = self.getClusterCentures(
            img_paths=img_paths, num_words=self.num_words, dataset_matrix=None)
        matrix = np.array(des_list)
        np.save(os.path.join(self.method, 'features_bases.npy'), matrix)
        np.save(os.path.join(self.method, 'centres.npy'), np.array(centres))
        img_features = self.get_all_features(
            des_list=des_list, num_words=num_words, centres=centres)
        np.save(os.path.join(self.method, 'code_bases.npy'),
                np.array(img_features))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='sift',
                        choices=['sift', 'surf', 'orb', 'test'])
    config = parser.parse_args()
    train_dataset_path = './ImageBase/train'
    num_words = 12
    sift_train = Train(dataset_path=train_dataset_path,
                       num_words=num_words, method=config.method)
    sift_train.train()
