import numpy as np
import cv2


input_mat = np.float32(np.random.randn(10,4))
search_mat = np.float32(np.random.randn(1,4))
input_mat_1 = np.float32(np.random.randn(5, 4))
print(input_mat)
print(input_mat_1)
print(search_mat)
FLANN_INDEX_KDTREE=1

#从本地加载模型
params=dict(algorithm=FLANN_INDEX_KDTREE,trees=1)
def read_yml():
    fs = cv2.FileStorage("tree.yml", cv2.FILE_STORAGE_READ)
    fn = fs.getNode("tree")
    print(fn.mat())
#建树
kdtree=cv2.flann_Index()
kdtree.build(input_mat,params)
kdtree.build(input_mat_1,params)
print("build tree success")
#检索
indices, dists=kdtree.knnSearch(search_mat,2,params=-1)
print("search kdtree")
#输出检索结果
print(indices)
print(np.sqrt(dists))
#使用numpy下的欧式距离验证结果是否正确
print((np.linalg.norm(input_mat-search_mat,axis=1)))
#模型
f=cv2.FileStorage("tree.yml",cv2.FILE_STORAGE_WRITE)
f.write("tree",input_mat)
f.release()   