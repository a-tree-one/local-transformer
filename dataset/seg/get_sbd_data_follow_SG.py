import torch
import torch.utils.data as Data
import numpy as np
from PIL import Image
import json
from torchvision.transforms import transforms
from dataset.dataset_val.dataset_sbd import DBInterface


class MyDataset_val(Data.Dataset):
    '''
    To sample 1000 pairs of images for validattion, the code follows "SG-One: Similarity guidance network
    for one-shot semantic segmentation"
    '''
    def __init__(self, test_group, k_shot):
        super(MyDataset_val, self).__init__()

        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([transforms.Resize((320, 320))])

        #follow SG-One to set the test_group and k-shot
        profile_set = 'fold%d_%dshot_test' % (test_group, k_shot)
        self.params = {
            'profile': profile_set,
            'second_label_params': [('first_label', 1.0, 0.0)],
            'first_label_params': [('second_label', 1.0, 0.0)],
            'batch_size': 1,
            'k_shot': k_shot,
            'has_cont': False,
            'deploy_mode': True,  #
        }

        if True:
            # settings = __import__('.ss_settings')
            import dataset.dataset_val.ss_settings as settings
            profile = getattr(settings, self.params['profile'])
            profile.update(self.params)  ###profile是一个Map对象，根据新的参数更新原来的
            params = profile

        self.db_interface = DBInterface(params)

        self.len_of_dataset = params.db_cycle

        self.data_list = []

        for num_data in range(self.len_of_dataset):
            data_temp = self.db_interface.next_pair()
            self.data_list.append(data_temp)

    def __len__(self):
        return self.len_of_dataset

    def __getitem__(self, index):
        data_temp = self.data_list[index]
        first_index = data_temp[1][0]
        second_index = data_temp[2]
        class_index = int(data_temp[0].name[-2:])
        class_image_subset = data_temp[0].video_item.image_items
        first_image_item = class_image_subset[first_index]
        second_image_item = class_image_subset[second_index]

        first_image_item_ori_path = first_image_item.img_path
        first_image_item_gt_path = first_image_item.mask_path

        second_image_item_ori_path = second_image_item.img_path
        second_image_item_gt_path = second_image_item.mask_path

        image_pair_path_list = [first_image_item_ori_path, second_image_item_ori_path]
        gt_pair_path_list = [first_image_item_gt_path, second_image_item_gt_path]

        image_list, gt_list, name_list, label_pair = self.convert2tensor(image_pair_path_list,
                                                                         gt_pair_path_list, class_index)

        return image_list, gt_list, name_list, label_pair

    def convert2tensor(self, img_pair, gt_pair, pair_label):
        '''
        :param img_pair: the path of images pairs
        :param gt_pair: the path of ground-truth
        :param pair_label: the class label
        :return:
        '''
        img_list = []
        gt_list = []
        name_list = []
        for img_num in range(len(img_pair)):
            img_file = img_pair[img_num]
            img = Image.open(img_file).convert('RGB')
            gt_file = gt_pair[img_num]
            name = img_file.split('/')[-1].split('.')[0]
            gt = Image.open(gt_file).convert('P')

            # img.show()
            # gt.show()

            if self.transform is not None:
                img = self.transform(img)
                gt = self.transform(gt)
            img = np.array(img, dtype=np.uint8)
            gt = np.array(gt)

            gt = gt.astype(np.int64)
            gt_ori = gt.copy()
            gt[gt != pair_label] = 0
            gt[gt == pair_label] = 1
            gt[gt_ori == 255] = 0

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
        return img_list, gt_list, name_list, pair_label


if __name__ == '__main__':

    json_path = '/media/yyw/JX_disk/yyw_disk/datasets/sbd/json_val_ori/val_1_shot_fold3.json'
    dataset = MyDataset_val(test_group=3, k_shot=1)
    # print(dataset[1000][1])
    print(len(dataset))
    print('done')

    json_list = []

    train_loader = Data.DataLoader(dataset=dataset, batch_size=64, shuffle=False, num_workers=2)
    for step, (x, gt, name, label) in enumerate(train_loader):
        # torchvision.utils.save_image(x, '/media/yyw/JX_disk/yyw_disk/yyw/pycharm_pro/feature_ortho/experiment.1.jpg')

        for ii in range(x[0].size()[0]):
            dict_data = dict(query_image=name[1][ii], support_image=name[0][ii], label=int(label[ii]))
            json_list.append(dict_data)

        print(step)
    with open(json_path, 'w') as f:
        json.dump(json_list,f)


