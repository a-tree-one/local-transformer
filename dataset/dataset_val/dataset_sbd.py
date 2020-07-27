import random
import copy

from .util import cprint, bcolors
from .util import DBImageItem, ImagePlayer, VideoPlayer, change_coordinates
from .pascal_sbd import PASCAL, PASCAL_READ_MODES


class DBInterface():
    def __init__(self, params):
        self.params = params  ###传入的参数，　数据集的形式，分组之类的信息
        self.load_items()

        # initialize the random generator
        self.init_randget(params['read_mode'])
        self.cycle = 0

    def init_randget(self, read_mode):
        '''
        根据read_mode, 选择读取图片的方式，　deterministic/shuffle
        :param read_mode:
        :return:
        '''
        self.rand_gen = random.Random()
        if read_mode == 'shuffle':
            self.rand_gen.seed()
        elif read_mode == 'deterministic':
            self.rand_gen.seed(1385)  # >>>Do not change<<< Fixed seed for deterministic mode.

    def update_seq_index(self):
        '''
        :return:
        '''
        self.seq_index += 1
        if self.seq_index >= len(self.db_items):
            self.db_items = copy.copy(self.orig_db_items)
            self.rand_gen.shuffle(self.db_items)
            self.seq_index = 0

    def next_pair(self):
        end_of_cycle = 'db_cycle' in self.params.keys() and self.cycle >= self.params['db_cycle']
        if end_of_cycle:
            assert (self.params['db_cycle'] > 0)
            self.cycle = 0
            self.seq_index = len(self.db_items)
            self.init_randget(self.params['read_mode'])

        self.cycle += 1
        base_trans = None if self.params['image_base_trans'] is None else self.params['image_base_trans'].sample()
        self.update_seq_index()
        if self.params['output_type'] == 'single_image':
            db_item = self.db_items[self.seq_index]
            assert (isinstance(db_item, DBImageItem))
            player = ImagePlayer(db_item, base_trans, None, None, length=1)
            return player, [0], None
        elif self.params['output_type'] == 'image_pair':
            imgset, second_index = self.db_items[self.seq_index]
            player = VideoPlayer(imgset, base_trans, self.params['image_frame_trans'])
            set_indices = list(range(second_index)) + list(range(second_index + 1, player.length))
            assert (len(set_indices) >= self.params['k_shot'])
            self.rand_gen.shuffle(set_indices)
            first_index = set_indices[:self.params['k_shot']]
            return player, first_index, second_index
        else:
            raise Exception('Only single_image and image_pair mode are supported')

    def _remove_small_objects(self, items):
        '''
        移除一些目标较小的目标
        :param items: 以DBPascalItem为数据格式list
        :return: 移除小目标后的list,　元素任然是DBPascalItem
        '''
        filtered_item = []
        for item in items:
            mask = item.read_mask()
            if change_coordinates(mask, 32.0, 0.0).sum() > 2:
                filtered_item.append(item)
        return filtered_item

    ###应该是根据传入的参数导入对应的数据，并将其放到self.db_items中
    def load_items(self):
        self.db_items = []
        if True:
            for image_set in self.params['image_sets']:
                print(image_set)  ###选择对应的数据集，　就像pascal_traing
                if image_set.startswith('pascal') or image_set.startswith('sbd'):
                    if image_set.startswith('pascal'):
                        pascal_db = PASCAL(self.params['pascal_path'],
                                                image_set[7:])  ###根据对应的数据集和trainning/val选择数据集
                    elif image_set.startswith('sbd'):
                        pascal_db = PASCAL(self.params['sbd_path'], image_set[4:])
                        print('sbd_dataset is built')
                        # reads single image and all semantic classes are presented in the label

                    if self.params['output_type'] == 'single_image':
                        items = pascal_db.getItems(self.params['pascal_cats'],
                                                   read_mode=PASCAL_READ_MODES.SEMANTIC_ALL)
                    # reads pair of images from one semantic class and and with binary labels
                    elif self.params['output_type'] == 'image_pair':
                        items = pascal_db.getItems(self.params['pascal_cats'],
                                                   read_mode=PASCAL_READ_MODES.SEMANTIC)
                        print('sbd_dataset is loaded')
                        if image_set[4:] == 'test':
                            items = self._remove_small_objects(items)
                    else:
                        raise Exception('Only single_image and image_pair mode are supported')
                    self.db_items.extend(items)
                else:
                    raise Exception
            cprint('Total of ' + str(len(self.db_items)) + ' db items loaded!', bcolors.OKBLUE)

            # reads pair of images from one semantic class and and with binary labels
            if self.params['output_type'] == 'image_pair':
                items = self.db_items

                clusters = PASCAL.cluster_items(self.db_items)

                self.db_items = []
                for item in items:
                    # print(item)
                    set_id = item.obj_ids[0]
                    imgset = clusters[set_id]
                    assert (imgset.length > self.params[
                        'k_shot']), 'class ' + imgset.name + ' has only ' + imgset.length + ' examples.'
                    in_set_index = imgset.image_items.index(item)
                    self.db_items.append((imgset, in_set_index))
                cprint('Total of ' + str(len(clusters)) + ' classes!', bcolors.OKBLUE)

        self.orig_db_items = copy.copy(self.db_items)  ###深度复制一份原来序列的数据

        assert (len(self.db_items) > 0), 'Did not load anything from the dataset'
        self.seq_index = len(self.db_items)
