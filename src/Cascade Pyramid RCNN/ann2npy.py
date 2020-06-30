import os
import pcl
import numpy as np
import joblib

# import torch

# class Point(object):
#     def __init__(self,x,y,z):
#         self.x = x
#         self.y = y
#         self.z = z
# points = []

# pathbox = 'D:/JupyterNotebook/testset/_bbox/'
# pathpcd = 'D:/JupyterNotebook/testset/pcd/'
origin_path='/media/tonyw/SSD/Workspace/JupyterNotebook/testset/'
# pathbox = '/media/tonyw/SSD/Workspace/JupyterNotebook/testset_new/test_new/bbox/'
# pathbox = '/media/tonyw/SSD/Workspace/JupyterNotebook/testset_new/train_new/bbox/'
# origin_path='D:/Workspace/JupyterNotebook/new_data/'
# pathbox = '/media/tonyw/SSD/Workspace/JupyterNotebook/new_data/test_1/bbox/'
pathbox = '/media/tonyw/SSD/Workspace/JupyterNotebook/new_data/train_1/bbox/'
max_target_num = 12  #<=12
# print('===')

for fpathe, dirs, fs in os.walk(pathbox):
    # print(fpathe,dirs,fs)
    cloudata = []
    annotationdata = []
    uni = []
    totalcount = 0
    countff = 0
    # print(pathbox)
    time_stamps=[]

    for f in fs:
        # print(os.path.join(fpathe, f))
        print(f)
        pcdname = f.replace('.txt', '.pcd')
        time_stamp = f.split('.')[0] + '.' + f.split('.')[1]
        # print(pcdname)
        p = pcl.PointCloud()
        p.from_file(origin_path+'pcd/' + pcdname)
        parray = p.to_array()
        parray = np.delete(parray, -1, axis=-1)
        cloudata.append(parray)
        time_stamps.append(time_stamp)

        # with open(pathbox + f, 'r')as ff:
        with open(origin_path+'_bbox/' + f, 'r')as ff:
            bboxarray = []
            uniboxes = np.zeros((12, 5), dtype=np.float)
            count = 0
            countff += 1
            for i, line in enumerate(ff):
                bboxlabel = line.strip().split()
                bboxlabel = np.delete(bboxlabel, (0, 1, 4, 7, 8), axis=0)
                bboxlabel = map(float, bboxlabel)
                uniboxes[i] = bboxlabel
                count += 1
            totalcount += count
            b = np.zeros((12, 1), dtype=np.float)
            data = np.c_[uniboxes, b]
            data[:, -1] = data[:, 0] * data[:, 1]  # sorted by box area
            data = data[(data[:, -1].argsort())[::-1]]  # descending order
            # print(data)
            fiveboxes = data[0:max_target_num , 0:5]  # Half open, half closed interval!!
            # print(fiveboxes)

            # print(bboxlabel)
            annotationdata.append(fiveboxes)  # first column aligned between pcd and annotation
    print(totalcount, countff)
    # cloudtensor=torch.FloatTensor(cloudata)
    cloudata = np.array(cloudata, dtype=np.float)
    annotationdata = np.array(annotationdata, dtype=np.float)
    print(cloudata.shape)
    print(annotationdata.shape)
    # np.save('D:/JupyterNotebook/testset/cloudata', cloudata)
    # np.save('D:/JupyterNotebook/testset/anndata', annotationdata)
    # np.save('/media/tonyw/SSD/Workspace/JupyterNotebook/testset/cloudata_test_new', cloudata)
    # np.save('/media/tonyw/SSD/Workspace/JupyterNotebook/testset/anndata_test_new', annotationdata)
    np.save('/media/tonyw/SSD/Workspace/JupyterNotebook/testset/cloudata_train_new', cloudata)
    np.save('/media/tonyw/SSD/Workspace/JupyterNotebook/testset/anndata_train_new', annotationdata)

#
# a=0
# for ann in annotationdata:
#     l=len(ann)
#     if l>a:
#         a=l
# print("========",a)


# filename = 'D:/1544600733.580758018'
#
# p = pcl.PointCloud()
# p.from_file(filename+'.pcd')
# print(p)
# p.to_file("ppp.txt")
# a=p.to_array()
# print(a)
# with open(filename+'.pcd','r',encoding='UTF-8') as f:
#     for line in  f.readlines()[11:len(f.readlines())-1]:
#         strs = line.split(' ')
#         points.append(Point(strs[0],strs[1],strs[2].strip()))

# fw = open(filename+'.txt','w')
# for i in range(len(points)):
#      linev = points[i].x+" "+points[i].y+" "+points[i].z+"\n"
#      fw.writelines(linev)
# fw.close()
