import numpy as np
from PIL import Image
import os
import cv2
import argparse
from utils import *
import time

ANIMALS = {'cat':0, 'cow':1, 'tiger':2, 'rabbit':3, 'dragon':4, 'snake':5,
           'horse':6, 'goat':7, 'monkey':8, 'chicken':9, 'dog':10, 'pig':11}


class Test():
    def __init__(self, feature_base_path, centers_path, img_paths_path, num_words, num_close, method, encode):
        self.num_words = num_words
        self.num_close = num_close
        self.method = method
        self.encode = encode
        self.feature_base_path = os.path.join(self.method, feature_base_path)
        self.centers_path = os.path.join(self.method, centers_path)
        self.img_paths_path = os.path.join(self.method, img_paths_path)

        if os.path.exists(self.feature_base_path):
            self.feature_base = np.load(self.feature_base_path)
        if os.path.exists(self.centers_path):
            self.centers = np.load(self.centers_path)
        if os.path.exists(self.img_paths_path):
            self.img_paths = np.load(self.img_paths_path)

        if self.method == 'sift':
            self.feat_extract = cv2.xfeatures2d.SIFT_create()
            self.feature_num = 128
        elif self.method == 'surf':
            self.feat_extract = cv2.xfeatures2d.SURF_create()
            self.feature_num = 64
        elif self.method == 'orb':
            self.feat_extract = cv2.ORB_create(1000)
            self.feature_num = 32

    def getNearestImg(self, feature, dataset, num_close):
        '''
        找出目标图像最像的几个
        feature:目标图像特征
        dataset:图像数据库
        num_close:最近个数
        return:最相似的几个图像
        '''
        features = np.ones((dataset.shape[0], len(feature)), 'float32')
        features = features*feature
        dist = np.sum((features-dataset)**2, 1)
        dist_index = np.argsort(dist)
        return dist_index[:num_close]

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

    '''
    def VLAD_des2feature(self, des, num_words, centures):
        FLANN_INDEX_KDTREE=1
        kdtree = cv2.flann_Index()
        params = dict(algorithm=FLANN_INDEX_KDTREE, trees=num_words)
        kdtree.build(des, params)
        return kdtree
    '''

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

    def draw_keypoints(self, img_name, save_dir):
        img = Image.open(img_name)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.array(img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # descriptor = self.feat_extract.SIFT_create()
        (kps, features) = self.feat_extract.detectAndCompute(gray, None)
        # kps = np.float32([kp.pt for kp in kps])
        cv2.drawKeypoints(img, kps, img, (255, 0, 0),
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.drawKeypoints(I,kps,I,(0,255,255))
        Image.fromarray(img).save(os.path.join(
            save_dir, self.method + '_keypoints.jpg'))

    def multikind_test(self, test_dir, kind):
        test_path = os.path.join(test_dir, kind)
        num = len(os.listdir(test_path))
        print(num)
        kind_result = np.zeros(num)
        result = np.zeros(num)
        for img_index, name in enumerate(os.listdir(test_path)):
            if name.split('.')[-1] != 'jpg':
                continue
            img_name = os.path.join(test_path, name)
            img_ID = name.split('.')[-2]

            print('{} is retrievaling ......'.format(img_name))
            img = Image.open(img_name)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Rotate
            # img = Rotate(img)
            img = Scale(img)

            img = np.array(img)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            kp, des = self.feat_extract.detectAndCompute(gray, None)

            if self.encode == 'bow':
                # (1, self.num_words)
                feature = self.des2feature(
                    des=des, centures=self.centers, num_words=self.num_words)
                sorted_index = self.getNearestImg(
                    feature, self.feature_base, self.num_close)
            elif self.encode == 'vlad':
                # (self_num_words x self.feature_num, 1)
                feature = self.VLAD_des2feature(
                    des=des, centures=self.centers, num_words=self.num_words)

            save_dir = os.path.join(test_dir, kind + '_result', name.split('.')[
                                    0], self.method)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # 取排列第一的图的种类作为预测的种类
            the_first_path = self.img_paths[sorted_index[0]]
            kind_pred = the_first_path.split('\\')[-2]
            kind_result[img_index] = ANIMALS[kind_pred]
            
            
            for index, i in enumerate(sorted_index):
                path = self.img_paths[i]
                
                kind_pred = path.split('\\')[-2]
                if kind_pred == kind:
                    imID_pred = path.split('\\')[-1].split('.')[-2]
                    if imID_pred == img_ID:
                        result[img_index] = 1
                

                print('\t{} has been retrievaled'.format(path))
                img = cv2.imread(path)
                save_path = os.path.join(save_dir, str(index) + '.jpg')
                cv2.imwrite(save_path, img)
            
            self.draw_keypoints(img_name, save_dir)
            
        
        print(result)
        precision = sum(result) / num
        '''
        label = np.ones(num) * ANIMALS[kind]
        tp_result = np.zeros(num)
        tp_result[(kind_result - label) == 0] = 1
        precision = sum(tp_result) / num
        '''
        return precision

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='seen',
                        choices=['seen', 'unseen'])
    parser.add_argument('--method', type=str, default='sift',
                        choices=['sift', 'surf', 'orb'])
    parser.add_argument('--encode', type=str, default='bow',
                        choices=['bow', 'vlad'])
    parser.add_argument('--num_close', type=int, default=5)
    config = parser.parse_args()
    test_path = './ImageBase/test/' + config.mode
    test = Test(feature_base_path='code_bases.npy',
                centers_path='centres.npy', img_paths_path='image_paths.npy',
                num_words=12, num_close=config.num_close, method=config.method, encode=config.encode)
    precisions = {}
    begin_time = time.time()
    for kind in ANIMALS.keys():
        #test_path = os.path.join(test_path, kind)
        precision = test.multikind_test(test_path, kind)
        precisions[kind] = precision
    end_time = time.time()
    run_time = end_time - begin_time
    print(f'run time: {run_time}')
    for kind, pre in precisions.items():
        print(f'{kind} precision :  {pre}')
    mean_precision = np.array(list(precisions.values())).mean()
    print(f'mean precision :　{mean_precision}')
