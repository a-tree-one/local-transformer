# coding=utf-8
import torch
import torch.nn as nn
import os
import numpy as np
from PIL import Image
import argparse
import torch.backends.cudnn as cudnn
import glob
import copy
import torch.nn.functional as F
import cv2

from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax
import pydensecrf.densecrf as dcrf

from dataset.seg.pascal_seg_5shot import MyDataset_pair
from dataset.seg.get_sbd_data_follow_SG_5shot import MyDataset_val
from models.model_5shot import Model


from utils.loss_optim_tools import check_dir
from utils.loss_optim_tools import adjust_learning_rate
from utils.train_tools import delete_existed_params
from utils.train_tools import AverageMeter
from utils.val_tools import index2color
from utils.val_tools import caculate_miou
from utils.val_tools import Cal_mIoU


parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-4, type=float, help="initial learning rate")
parser.add_argument('--weight_decay', default=1e-6, type=float, help="weight decay in optimizer")
parser.add_argument('--epoches', default=10000, type=float, help="the total epoch number of train")
parser.add_argument('--experiment_dir', default='/media/yyw/JX_disk/yyw_disk/yyw/pycharm_pro/2019late/Local-Transformer/'
                                                'experiment/Model_5shot_3')
parser.add_argument('--model_dir', default='/media/yyw/JX_disk/yyw_disk/yyw/pycharm_pro/2019late/Local-Transformer/'
                                           'experiment/Model_5shot_3/model_save')
parser.add_argument('--b', type=int, default=6)    # batch size
parser.add_argument('--b_v', type=int, default=1)    # batch size
parser.add_argument('--start_epoch', default=0, type=int, help="the epoch which start to train")
args = parser.parse_args()
print(args)


device_ids = [0, 0]

gt_dir = '/media/yyw/JX_disk/yyw_disk/datasets/sbd/cls'
img_dir = '/media/yyw/JX_disk/yyw_disk/datasets/sbd/img'


def crf(img, scores):
    '''
    scores: prob numpy array after softmax with shape (_, C, H, W)
    img: image of shape (H, W, C)
    CRF parameters: bi_w = 4, bi_xy_std = 121, bi_rgb_std = 5, pos_w = 3, pos_xy_std = 3
    '''

    img = np.asarray(img, dtype=np.uint8) # image.shape = [366,500,3]
    scores = np.asarray(scores.cpu().detach(), dtype=np.float32)

    scores = np.ascontiguousarray(scores)
    img = np.ascontiguousarray(img)

    n_classes, h, w = scores.shape

    d = dcrf.DenseCRF2D(w, h, n_classes)
    U = -np.log(scores)
    U = U.reshape((n_classes, -1))
    d.setUnaryEnergy(U)

    feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.FULL_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
    feats = create_pairwise_bilateral(sdims=(40, 40), schan=(7, 7, 7),
                                      img=img, chdim=2)
    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.FULL_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)

    Q = np.array(Q).reshape((2, h, w))

    for ii in range(Q.shape[0]):
        Q[ii] = cv2.medianBlur(Q[ii], 5)
        # Q[ii] = cv2.GaussianBlur(Q[ii], (7, 7), 1)

    Q_score = copy.deepcopy(Q)
    Q = np.argmax(Q, axis=0)

    return Q, Q_score


def setup_experiment_dirs():
    check_dir(args.experiment_dir)
    check_dir(args.model_dir)
    args.log_path = os.path.join(args.experiment_dir, 'log.txt')


def get_dataloader_few_shot():
    train_loader = torch.utils.data.DataLoader(MyDataset_pair(test_group=3),
        batch_size=args.b, shuffle=True, num_workers=0, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        MyDataset_val(test_group=3, k_shot=5),
        batch_size=int(args.b_v), shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader


def get_model():

    model = Model()
    if torch.cuda.is_available():
        model = model.to(device=device_ids[0])
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    saves = os.listdir(args.model_dir)  # todo save all info while train and val. so that can get info when retrain
    if len(saves) != 0:
        pre_epoch = max([eval(item.split(':_')[1].split('_')[0]) for item in saves])
        model_params_path = glob.glob('{}/*_{}_*.pkl'.format(args.model_dir, pre_epoch))[0]
        print(model_params_path)
        model.load_state_dict(torch.load(model_params_path))
        args.start_epoch = pre_epoch + 1
        args.best_val = eval(model_params_path.split('perform:_')[1].split('_train')[0])
        args.best_train = 0
    else:
        args.best_train = 0
        args.best_val = 0
    cudnn.benchmark = True
    return model


def get_criterion():
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    return criterion


def get_reconstruction_loss():
    criterion = nn.MSELoss()
    return criterion


def get_optimizer_pair(model):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr, amsgrad=True, weight_decay=args.weight_decay)
    return optimizer


def save_batch_gt(predict, name_list, label_list, batch_index):
    # outputs = predict.data.cpu().numpy()

    predict_list = []
    gt_list = []
    for ii, (msk, name, label) in enumerate(zip(predict, name_list, label_list)):
        label = int(label)
        gt_path = gt_dir + '/' + name+'.png'
        img_path = img_dir + '/' + name+'.jpg'
        gt = Image.open(gt_path).convert('P')
        img = Image.open(img_path).convert('RGB')

        # img = np.array(img, dtype=np.uint8)

        w, h = gt.size
        # w, h = 320, 320

        predict_temp = predict[ii]
        predict_temp = predict_temp.unsqueeze(0)
        predict_temp = F.interpolate(predict_temp, (h, w), mode='bilinear')
        predict_temp = predict_temp.squeeze(0)
        # predict_temp = predict_temp.cpu().numpy()

        # for ii in range(predict_temp.shape[0]):
        #     predict_temp[ii] = cv2.GaussianBlur(predict_temp[ii], (7, 7), 0)

        predict_temp, _ = crf(img, predict_temp)

        # _, predict_temp = torch.max(predict_temp, dim=0)
        # predict_temp = predict_temp.squeeze()
        # predict_temp = predict_temp.data.cpu().numpy()
        predict_list.append(predict_temp)

        output_img = np.zeros((h, w, 3), dtype=np.uint8)
        for i, color in enumerate(index2color):
            output_img[predict_temp == i, :] = color
        # output_img = np.resize(output_img, (320, 320))
        output_img = Image.fromarray(output_img)
        check_dir(args.experiment_dir + '/_vis_val_que')
        output_img.save('{}/{}.png'.format(args.experiment_dir
                                           + '/_vis_val_que', str(batch_index * args.b_v + ii)
                                           + '_pre', 'PNG'))

        gt = np.array(gt)
        gt = gt.astype(np.int64)
        gt_ori = gt.copy()
        gt[gt != label] = 0
        gt[gt == label] = 1
        gt[gt_ori == 255] = 0
        gt_list.append(gt)

    return predict_list, gt_list


class Trainer():
    def __init__(self, model=None, criterion=None, flag_val=None, break_for_debug=True):
        self.model = model
        self.criterion = criterion

        self.re_loss = get_reconstruction_loss()
        # self.optimizer_fcn = get_optimizer_fcn(self.model)
        self.optimizer_pair = get_optimizer_pair(self.model)

        self.train_loader_pair, self.val_loader_pair = get_dataloader_few_shot()
        self.break_for_debug = break_for_debug
        self.flag_val = flag_val
        self.current_epoch = 0
        self.val_performence_current_epoch = 0
        self.train_performence_current_epoch = 0
        self.best_val = 0
        self.start()
        self.miou = 0
        # self.guass_net = Gauss_Net()

    def start(self):
        if self.flag_val == 'just_train':  ## 一直训练
            pass
        if self.flag_val == 'just_val':  # 仅仅val一次　
            self.val_pair()
        if self.flag_val == 'train_val':  # 每次训练后val一次
            for epoch in range(args.start_epoch, args.epoches):
                adjust_learning_rate(self.optimizer_pair, epoch)
                self.current_epoch = epoch

                self.train_pair()
                self.val_pair()

                if self.val_performence_current_epoch > self.best_val:
                    self.best_val = self.val_performence_current_epoch
                    model_params_save_path = os.path.join(args.model_dir, 'epoch:_{}_perform:_{}_{}.pkl'
                                                          .format(self.current_epoch,
                                                                  self.val_performence_current_epoch,
                                                                  self.flag_val))
                    torch.save(self.model.state_dict(), model_params_save_path)
                    delete_existed_params(args.model_dir)
                    print('\nnew best val saved epoch {} best val:{} '.format(self.current_epoch,
                                                                              self.val_performence_current_epoch))
                    string = '{}   {}  {}　{}'.format(self.flag_val, self.current_epoch, self.val_performence_current_epoch, self.miou)
                    with open(args.log_path, 'a+') as f:
                        f.write(string + '\n')
                # adjust_learning_rate(self.optimizer, epoch, args.epochs, args.lr)
        # self.writer.close()

    def get_R_truth(self, gt_que_1, gt_sup_1):
        gt_que_1 = gt_que_1.unsqueeze(1)
        gt_que_1 = F.interpolate(gt_que_1, (20, 20), mode='bilinear')
        gt_que_1 = gt_que_1.view(gt_que_1.size()[0], -1, 1)

        gt_sup_1 = gt_sup_1.unsqueeze(1)
        gt_sup_1 = F.interpolate(gt_sup_1, (20, 20), mode='bilinear')
        gt_sup_1 = gt_sup_1.view(gt_sup_1.size()[0], 1, -1)
        R_truth = torch.matmul(gt_que_1, gt_sup_1)
        return R_truth

    def train_pair(self):
        loss_1_epoch = AverageMeter()
        loss_2_epoch = AverageMeter()
        loss_3_epoch = AverageMeter()
        loss_4_epoch = AverageMeter()
        miou_epoch = AverageMeter()
        miou_fore_epoch = AverageMeter()
        self.model.train()
        for batch_index, (data_list, gt_list, name_list) in enumerate(self.train_loader_pair):
            if self.break_for_debug:
                if batch_index == 5:
                    break
            data_sup_list = data_list[:5]
            data_que = data_list[5]

            gt_sup_list = gt_list[:5]
            gt_que = gt_list[5]

            data_sup_list = [data_sup_1.to(device=device_ids[0]) for data_sup_1 in data_sup_list]
            data_que = data_que.to(device=device_ids[0])

            gt_sup_list = [gt_sup_1.to(device=device_ids[0]) for gt_sup_1 in gt_sup_list]
            gt_que = gt_que.to(device=device_ids[0])

            gt_sup_float_list = [gt_sup_1.type(torch.FloatTensor).to(device=device_ids[0]) for gt_sup_1 in gt_sup_list]
            gt_que_float = gt_que.type(torch.FloatTensor).to(device=device_ids[0])

            output = self.model(x_q=data_que, x_s_list=data_sup_list, x_s_mask_list=gt_sup_float_list, is_train=True)

            seg_q, att_mask, R_list, seg_q_fake_list, att_mask_fake_list, R_list_fake_list = \
                output[0], output[1], output[2], output[3], output[4], output[5]

            loss_1 = self.criterion(seg_q, gt_que) + self.criterion(att_mask, gt_que)

            loss_2 = 0
            for n_s in range(5):
                gt_que_1 = gt_que_float
                gt_sup_1 = gt_sup_float_list[n_s]

                R_truth_temp = self.get_R_truth(gt_que_1, gt_sup_1)
                loss_2_temp = self.re_loss(R_list[n_s], R_truth_temp)
                loss_2 = loss_2 + loss_2_temp

            loss_3 = 0
            loss_4 = 0
            for n_s in range(5):
                fake_que_att_mask = att_mask_fake_list[n_s]
                fake_que_seg = seg_q_fake_list[n_s]
                fake_R_list =R_list_fake_list[n_s]

                gt_sup_list_copy = copy.deepcopy(gt_sup_list)
                gt_sup_float_list_copy = copy.deepcopy(gt_sup_float_list)

                fake_que_gt = gt_sup_list_copy[n_s]
                fake_que_gt_float = gt_sup_float_list_copy[n_s]

                loss_3_temp = self.criterion(fake_que_att_mask, fake_que_gt) + self.criterion(fake_que_seg, fake_que_gt)

                loss_3 = loss_3 + loss_3_temp

                for n_s_s in range(5):
                    gt_que_1 = fake_que_gt_float
                    gt_sup_1 = gt_sup_float_list_copy[n_s_s]

                    R_truth_temp = self.get_R_truth(gt_que_1, gt_sup_1)
                    loss_4_temp = self.re_loss(fake_R_list[n_s_s], R_truth_temp)
                    loss_4 = loss_4 + loss_4_temp

            loss = loss_1 + loss_2 + loss_3 + loss_4

            self.optimizer_pair.zero_grad()
            loss.backward()
            self.optimizer_pair.step()

            _, predict = torch.max(seg_q, dim=1)

            predict_temp = predict.cpu().data.numpy()
            gt_temp = gt_que.cpu().data.numpy()
            miou, miou_fore = caculate_miou(predict_temp, gt_temp, 2)
            loss_1_epoch.update(loss_1.data.item())
            loss_2_epoch.update(loss_2.data.item())
            loss_3_epoch.update(loss_3.data.item())
            loss_4_epoch.update(loss_4.data.item())

            miou_epoch.update(miou)
            miou_fore_epoch.update(miou_fore)

            print('train:\t{}|{}\t{}|{}\tloss_1:{}\tloss_2:{}\tloss_3:{}\tloss_4:{}\tmiou:{}\tmiou_fore:{}'.format(
                                                                        self.current_epoch, args.epoches,
                                                                        batch_index + 1, len(self.train_loader_pair),
                                                                        loss_1_epoch.avg, loss_2_epoch.avg,
                                                                        loss_3_epoch.avg, loss_4_epoch.avg,
                                                                        miou_epoch.avg, miou_fore_epoch.avg))

    def val_pair(self):
        loss_1_epoch = AverageMeter()
        self.model.eval()
        self.cal_miou = Cal_mIoU()
        with torch.no_grad():
            for batch_index, (data_list, gt_list, name_list, label_pair) in enumerate(self.val_loader_pair):
                if self.break_for_debug:
                    if batch_index == 5:
                        break
                data_sup_list = data_list[:5]
                data_que = data_list[5]

                gt_sup_list = gt_list[:5]
                gt_que = gt_list[5]

                name_que = name_list[5]

                data_sup_list = [data_sup_1.to(device=device_ids[0]) for data_sup_1 in data_sup_list]
                data_que = data_que.to(device=device_ids[0])

                gt_sup_list = [gt_sup_1.to(device=device_ids[0]) for gt_sup_1 in gt_sup_list]
                gt_que = gt_que.to(device=device_ids[0])

                gt_sup_float_list = [gt_sup_1.type(torch.FloatTensor).to(device=device_ids[0]) for gt_sup_1 in
                                     gt_sup_list]

                output = self.model(x_q=data_que, x_s_list=data_sup_list, x_s_mask_list=gt_sup_float_list, is_train=False)

                loss_1 = self.criterion(output[1], gt_que)

                _, predict = torch.max(output[1], dim=1)

                loss_1_epoch.update(loss_1.data.item())

                print('train:\t{}|{}\t{}|{}\tloss_1:{}\t'.format(self.current_epoch, args.epoches,
                                                                 batch_index + 1,
                                                                 len(self.val_loader_pair),
                                                                 loss_1_epoch.avg))
                # _, predict = torch.max(predict_2d, dim=1)
                outputs = predict.data.cpu().numpy()
                for ii, msk in enumerate(outputs):
                    sz = msk.shape[0]
                    output_img = np.zeros((sz, sz, 3), dtype=np.uint8)
                    for i, color in enumerate(index2color):
                        output_img[msk == i, :] = color
                    output_img = Image.fromarray(output_img)
                    check_dir(args.experiment_dir + '/_vis_val_que')
                    output_img.save('{}/{}.png'.format(args.experiment_dir
                                                       + '/_vis_val_que', str(batch_index*args.b_v+ ii)
                                                       + '_pre', 'PNG'))

                outputs = gt_que.data.cpu().numpy()
                for ii, msk in enumerate(outputs):
                    sz = msk.shape[0]
                    output_img = np.zeros((sz, sz, 3), dtype=np.uint8)
                    for i, color in enumerate(index2color):
                        output_img[msk == i, :] = color
                    output_img = Image.fromarray(output_img)
                    output_img.save('{}/{}.png'.format(args.experiment_dir + '/_vis_val_que',
                                                       str(batch_index*args.b_v+ ii) + '_gt_que'), 'PNG')

                predict_list, gt_list = save_batch_gt(predict=F.softmax(output[1], dim=1), name_list=name_que, label_list=label_pair, batch_index=batch_index)
                FB_IoU, fore_IoU = self.cal_miou.caculate_miou(predict_list, gt_list, 2)
                print('the total miou(on the original_size) is ', FB_IoU, fore_IoU)

            self.val_performence_current_epoch = fore_IoU

            string_1 = 'miou calculated on original---{}   {}  {}　{}'.format(self.flag_val, self.current_epoch,
                                                                             fore_IoU,
                                                                             FB_IoU)
            with open(args.log_path, 'a+') as f:
                f.write(string_1 + '\n')

            self.miou = fore_IoU


def main():
    setup_experiment_dirs()
    model = get_model()
    criterion = get_criterion()
    Trainer(model=model,
            criterion=criterion,
            flag_val='train_val',  # 这个用来控制仅仅训练，或者每个epoch后val一次,'just_train','just_val','train_val'
            break_for_debug=True)


if __name__ == '__main__':
    main()
