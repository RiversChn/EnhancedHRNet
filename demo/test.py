from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import cv2
import os

if __name__ == '__main__':

    annot_path_train = '/home/sihao/Desktop/COCO/annotations/person_keypoints_train2017.json'
    imgRoot = "../datasets/COCO"
    dataType = "train2017"
    coco = COCO(annot_path_train)

    imgId = coco.getAnnIds(imgIds=552272)
    imgInfo = coco.loadAnns(imgId)
    print(f'图像{imgId}的信息如下：\n{imgInfo}')

    imgId = coco.getImgIds(imgIds=552272)
    imgInfo = coco.loadImgs(imgId)[0]
    print(f'图像{imgId}的信息如下：\n{imgInfo}')

    imPath = os.path.join(imgRoot, dataType, imgInfo['file_name'])
    im = cv2.imread(imPath)
    plt.imshow(im);
    plt.axis('off')
    # plt.show()

    annIds = coco.getAnnIds(imgIds=imgInfo['id'])  # 获取该图像对应的anns的Id
    # print(f'图像{imgInfo["id"]}包含{len(anns)}个ann对象，分别是:\n{annIds}')

    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    print(f'ann{annIds[2]}对应的mask如下：')
    mask = coco.annToMask(anns[2])
    plt.imshow(mask);
    plt.axis('off')
    plt.show()
