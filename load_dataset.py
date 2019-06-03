import os
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
from skimage import io
from torch.utils.data import Dataset


class PascalVOCDataset(Dataset):
    """Pascal VOC dataset"""
    """
        input: data_dir
               year (2007, 2012)
               cls (all or one of the 20 classes)
               target (trainval or test)
        output: 1. image: nxmx3
                2. count: number of objects in the image
                3. label: dic<class, [[xmin, ymin, xmax, ymax]]>
    """
    def __init__(self, data_dir, year, cls, target, transform=None):
        self.data_dir = os.path.join(data_dir, 'VOCdevkit', 'VOC' + year)
        self.cls = cls
        self.image = []

        if cls == 'all':
            self.image = open(os.path.join(self.data_dir, 'ImageSets', 'Main', target + '.txt')).read().splitlines()
        else:
            with open(os.path.join(self.data_dir, 'ImageSets', 'Main', cls + '_' + target + '.txt'), 'r') as f:
                for line in f:
                    line = line.split()
                    if line[1] == '1':
                        self.image.append(line[0])

        self.transform = transform


    def __len__(self):
        return len(self.image)


    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, 'JPEGImages', self.image[idx] + '.jpg')
        img = io.imread(img_path)
        gt_box, gt_point = self.GenerateBoxandP(idx)
        transform_img = self.transform(img)
        sample = {'image': img, 'trans_img': transform_img, 'gt_box': gt_box, 'gt_point': gt_point}
        return sample


    def GenerateBoxandP(self, idx):
        anno_path = os.path.join(self.data_dir, 'Annotations', self.image[idx] + '.xml')
        tree = ET.parse(anno_path)
        root = tree.getroot()
        gt_box = {}
        gt_point = {}
        for obj in root.iter('object'):
            name = obj.find('name').text
            box = obj.find('bndbox')
            cor = [int(box.find('xmin').text), int(box.find('ymin').text),
                   int(box.find('xmax').text), int(box.find('ymax').text)]
            point = [(cor[0]+cor[2])/2, (cor[1]+cor[3])/2]
        
            if not name in gt_box:
                gt_box[name] = []
                gt_point[name] = []
            gt_box[name].append(cor)
            gt_point[name].append(point)
        return gt_box, gt_point