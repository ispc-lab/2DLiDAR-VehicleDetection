import numpy as np
from bBox_2D import bBox_2D


def GroundTruthSampling(clouddata, anndata):
    Pointset = []
    Annset = []
    idcount = 0
    for cc, scan in enumerate(clouddata):
        for aa, label in enumerate(anndata[cc]):

            if label[4] == -90 or label[4] == 90:
                box = bBox_2D(label[1], label[0], label[3], label[2], -label[4])  # fix annotations!!!
            else:
                box = bBox_2D(label[0], label[1], label[3], label[2], -label[4])  # clock wise

            if label[0] < 12 / 18 or label[1] < 12 / 18 or label[0] > 144 / 18 or label[1] > 144 / 18 or label[0] * \
                    label[1] < 360 / 324 or label[0] * \
                    label[1] > 13000 / 324:
                continue

            GTanns = {
                'scan_id': cc,
                'label_id': aa,
                'bbox': [box.xc, box.yc, box.width, box.length],
                'rotation': box.alpha,
                'id': idcount,
            }
            Annset.append(GTanns)

            pinbox = []
            box.resize(1.2)  # to ensure points inside
            box.bBoxCalcVertxex()
            maxx = max(box.vertex1[0], box.vertex2[0], box.vertex3[0], box.vertex4[0])
            minx = min(box.vertex1[0], box.vertex2[0], box.vertex3[0], box.vertex4[0])
            maxy = max(box.vertex1[1], box.vertex2[1], box.vertex3[1], box.vertex4[1])
            miny = min(box.vertex1[1], box.vertex2[1], box.vertex3[1], box.vertex4[1])
            for dot in scan:
                if minx < dot[1] < maxx and miny < dot[0] < maxy:  # vertexes in meters
                    pinbox.append([dot[1], dot[0]])  # while points in meters
            Pointset.append(np.array(pinbox))

            idcount += 1

    print(len(Pointset), len(Annset))
    np.save('./testset/GTpoints', Pointset)  # meters
    np.save('./testset/GTanns', Annset)  # meters

if __name__== '__main__':
    cloudata=np.load('./testset/cloudata_train_new.npy')
    annotationdata=np.load('./testset/anndata_train_new.npy', )
    GroundTruthSampling(cloudata,annotationdata)
