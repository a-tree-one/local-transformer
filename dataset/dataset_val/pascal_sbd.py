import numpy as np
import os.path as osp
from PIL import Image
import pickle
import json

from .util import cprint
from .util import bcolors
from .util import DBPascalItem
from .util import DBImageSetItem


class PASCAL_READ_MODES:
    #Returns list of DBImageItem each has the image and the mask for all semantic labels
    SEMANTIC_ALL = 1
    #Returns list of DBImageSetItem each has set of images and corresponding masks for each semantic label
    SEMANTIC = 2


class PASCAL:
    def __init__(self, db_path, dataType):
        '''
        :param db_path: 对应的数据集的路径
        :param dataType: 数据集的形式，是train/val
        '''
        if dataType == 'training':
            dataType = 'train'
        elif dataType == 'test':
            dataType = 'val'
        else:
            raise Exception('split \'' + dataType + '\' is not valid! Valid splits: training/test')

        self.db_path = db_path
        classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep',
                   'sofa', 'train', 'tv/monitor']
        self.name_id_map = dict(zip(classes, range(1, len(classes) + 1)))
        self.id_name_map = dict(zip(range(1, len(classes) + 1), classes))
        self.dataType = dataType

    def getCatIds(self, catNms=[]):  ###获得所选择的类别对应的ID号
        return [self.name_id_map[catNm] for catNm in catNms]

    def get_anns_path(self, read_mode):  ###获得对应的标注信息的地址，　但是.pkl不是模型的后缀吗？？？？貌似是pickle文件

        return osp.join(self.db_path, self.dataType + '_' + str(read_mode) + '_anns.json')

    def get_unique_ids(self, mask, return_counts=False, exclude_ids=[0, 255]):  ###根据语义分割的ＧＴ，得到对应的类别标签
        ids, sizes = np.unique(mask, return_counts=True)
        ids = list(ids)
        sizes = list(sizes)
        for ex_id in exclude_ids:
            if ex_id in ids:
                id_index = ids.index(ex_id)
                ids.remove(ex_id)
                sizes.remove(sizes[id_index])

        assert (len(ids) == len(sizes))
        if return_counts:
            return ids, sizes
        else:
            return ids

    ###将图片对应的路径，和gt对应的路径，以及类别标注写入到.pkl文件中
    def create_anns(self, read_mode):
        if self.db_path.endswith('pascal'):
           pass
        else:
            segmentation_sub_file = 'cls'
            with open(osp.join(self.db_path, self.dataType + '.txt'), 'r') as f:
                lines = f.readlines()
                names = []
                for line in lines:
                    if line.endswith('\n'):
                        line = line[:-1]
                    if len(line) > 0:
                        names.append(line)
        anns = []
        for item in names:
            mclass_path = osp.join(self.db_path, segmentation_sub_file, item + '.png')
            mclass_uint = np.array(Image.open(mclass_path))
            class_ids = self.get_unique_ids(mclass_uint)
            class_ids = [int(id) for id in class_ids]

            if read_mode == PASCAL_READ_MODES.SEMANTIC:
                for class_id in class_ids:
                    assert (class_id != 0 or class_id != 255)
                    anns.append(dict(image_name=item, mask_name=item, class_ids=[class_id]))
            elif read_mode == PASCAL_READ_MODES.SEMANTIC_ALL:
                anns.append(dict(image_name=item, mask_name=item, class_ids=class_ids))
            # print(dict(image_name=item, mask_name=item, class_ids=class_ids))
        with open(self.get_anns_path(read_mode), 'w') as f:
            json.dump(anns, f)

    ###获取对应的.pkl标注文件
    def load_anns(self, read_mode):
        path = self.get_anns_path(read_mode)
        if not osp.exists(path):
            self.create_anns(read_mode)
        with open(path, 'r') as f:
            anns = json.load(f)
        return anns

    ###根据所选择的类别对图片进行筛选
    def get_anns(self, catIds=[], read_mode=PASCAL_READ_MODES.SEMANTIC):
        anns = self.load_anns(read_mode)
        if catIds == []:
            return anns

        filtered_anns = []
        catIds_set = set(catIds)
        for ann in anns:
            class_inter = set(ann['class_ids']) & catIds_set
            # remove class_ids that we did not asked for (i.e. are not catIds_set)
            if len(class_inter) > 0:
                ann = ann.copy()
                ann['class_ids'] = sorted(list(class_inter))
                filtered_anns.append(ann)
        return filtered_anns

    ###获取对应包含所有的item
    def getItems(self, cats=[], read_mode=PASCAL_READ_MODES.SEMANTIC):
        if len(cats) == 0:  ###如果cats的长度为０，则默认为所有的类
            catIds = self.id_name_map.keys()
        else:
            catIds = self.getCatIds(catNms=cats)
        catIds = np.sort(catIds)

        anns = self.get_anns(catIds=catIds, read_mode=read_mode)
        cprint(str(len(anns)) + ' annotations read from pascal', bcolors.OKGREEN)

        items = []
        ids_map = None
        if read_mode == PASCAL_READ_MODES.SEMANTIC_ALL:
            old_ids = catIds
            new_ids = range(1, len(catIds) + 1)  ###这里需要将类别标签和gt_mask对用的标签重新进行映射一下，可能有的类别是不存在的就需要重新修改标签
            ids_map = dict(zip(old_ids, new_ids))
        for i in range(len(anns)):
            ann = anns[i]
            print(i, ann)
            img_path = osp.join(self.db_path, 'img', ann['image_name'] + '.jpg')

            mask_path = osp.join(self.db_path, 'cls', ann['mask_name'] + '.png')
            item = DBPascalItem('pascal-' + self.dataType + '_' + ann['image_name'] + '_' + str(i), img_path,
                                    mask_path, ann['class_ids'], ids_map)
            items.append(item)
        return items

    @staticmethod
    def cluster_items(items):
        clusters = {}  ###根据类别将数据存储在clusters这个字典中，不同的类别对应一个DBImageSetItem对象，其中包含了该类别多对应的所有图像
        for i, item in enumerate(items):
            assert (isinstance(item, DBPascalItem))
            item_id = item.obj_ids
            assert (len(item_id) == 1), 'For proper clustering, items should only have one id'
            item_id = item_id[0]
            # if clusters.has_key(item_id):
            if item_id in clusters.keys():
                clusters[item_id].append(item)
            else:
                clusters[item_id] = DBImageSetItem('set class id = ' + str(item_id), [item])
        return clusters
