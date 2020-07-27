import torch
import torch.utils.data as Data
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import json
import random
import matplotlib.pyplot as plt

json_train_data_path = '/media/yyw/JX_disk/yyw_disk/datasets/sbd/SBD_train_few_shot.json'
json_val_data_path = '/media/yyw/JX_disk/yyw_disk/datasets/sbd/SBD_val_few_shot.json'

random.seed(1234)

with open(json_train_data_path, 'r') as f:
    data_list_train = json.load(f)

with open(json_val_data_path, 'r') as f:
    data_list_val = json.load(f)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


class MyDataset_pair(Data.Dataset):
    def __init__(self, test_group=3):
        super(MyDataset_pair, self).__init__()
        self.list = []
        self.labels = []
        self.image_pair_path = []
        self.mask_gt_pair_path = []
        self.label_pair = []

        self.test_group = test_group
        self.image_cls_list = []
        self.gt_cls_list = []
        for ii in range(15):
            self.image_cls_list.append([])
            self.gt_cls_list.append([])
        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([transforms.Resize((320, 320))])
        self.data_list = data_list_train + data_list_val
        self.get_image_file()
        self.get_pair_images()

    def get_image_file(self):
        for i in range(len(self.data_list)):
            temp = self.data_list[i]
            label_temp_test = temp['cls_label'][self.test_group*5:self.test_group*5+5]

            label_temp_train = temp['cls_label'][:self.test_group*5] + temp['cls_label'][self.test_group*5+5:]

            label_flag = True
            for label in label_temp_test:
                if label == 1:
                    label_flag = False
                    break
            if label_flag:
                for num_label in range(len(label_temp_train)):
                    if label_temp_train[num_label] == 1:
                        self.image_cls_list[num_label].append(temp['img_path'])
                        self.gt_cls_list[num_label].append(temp['seg_gt_path'])

                self.list.append(temp['img_path'])
                self.labels.append(temp['seg_gt_path'])

    def get_pair_images(self):

        for num_cls in range(len(self.image_cls_list)):
            temp_image_cls_list = self.image_cls_list[num_cls]
            temp_gt_cls_list = self.gt_cls_list[num_cls]

            cls_image_sup = []
            cls_gt_sup = []
            cls_image_query = []
            cls_gt_query = []

            while len(temp_image_cls_list) > 6:
                temp_image_sup = []
                temp_gt_sup = []
                temp_image_query = []
                temp_gt_query = []
                for num_pair in range(6):
                    sample_num = random.sample(range(len(temp_image_cls_list)), 1)
                    sample_num = sample_num[0]
                    # print(sample_num, len(temp_image_cls_list))
                    if num_pair < 5:
                        temp_image_sup.append(temp_image_cls_list[sample_num])
                        temp_gt_sup.append(temp_gt_cls_list[sample_num])
                    else:
                        temp_image_query.append(temp_image_cls_list[sample_num])
                        temp_gt_query.append(temp_gt_cls_list[sample_num])

                    temp_image_cls_list.pop(sample_num)
                    temp_gt_cls_list.pop(sample_num)

                cls_image_sup.append(temp_image_sup)
                cls_gt_sup.append(temp_gt_sup)
                cls_image_query.append(temp_image_query)
                cls_gt_query.append(temp_gt_query)

            for ii in range(len(cls_image_sup)):
                for jj in range(len(cls_image_query)):
                    if ii == jj:
                        temp_image_path = cls_image_sup[ii] + cls_image_query[jj]
                        temp_gt_path = cls_gt_sup[ii] + cls_gt_query[jj]
                        self.image_pair_path.append(temp_image_path)
                        self.mask_gt_pair_path.append(temp_gt_path)
                        self.label_pair.append(num_cls)

    def __len__(self):
        return len(self.image_pair_path)

    def __getitem__(self, index):
        img_file_pair = self.image_pair_path[index]
        gt_file_pair = self.mask_gt_pair_path[index]
        img_list = []
        gt_list = []
        name_list = []
        mask_label_list = list(range(0, self.test_group*5)) + list(range(self.test_group*5+5, 20))
        pair_label = self.label_pair[index]
        for img_num in range(len(img_file_pair)):
            img_file = img_file_pair[img_num]
            img = Image.open(img_file).convert('RGB')
            gt_file = gt_file_pair[img_num]
            name = img_file.split('/')[-1].split('.')[0]
            gt = Image.open(gt_file).convert('P')

            if self.transform is not None:
                img = self.transform(img)
                gt = self.transform(gt)
            img = np.array(img, dtype=np.uint8)
            gt = np.array(gt)

            gt = gt.astype(np.int64)
            gt[gt == 255] = 0
            gt[gt != (mask_label_list[pair_label]+1)] = 0
            gt[gt == (mask_label_list[pair_label]+1)] = 1

            img = img.astype(np.float64) / 255
            if self.mean is not None:
                img -= self.mean
            if self.std is not None:
                img /= self.std

            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).float()
            gt = torch.from_numpy(gt)

            img_list.append(img)
            gt_list.append(gt)
            name_list.append(name)

        return img_list, gt_list, name_list


if __name__=='__main__':
    dataset = MyDataset_pair(test_group=3)
    # print(dataset[1000][1])
    print(len(dataset))
    print('done')

    train_loader = Data.DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=2)
    for step, (x, gt, name) in enumerate(train_loader):
        plt.imshow(gt[0][0])
        plt.show()

        plt.imshow(gt[1][0])
        plt.show()

        plt.imshow(gt[2][0])
        plt.show()

        plt.imshow(gt[3][0])
        plt.show()

        plt.imshow(gt[4][0])
        plt.show()

        plt.imshow(gt[5][0])
        plt.show()

        print("one sample")
