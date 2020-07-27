import numpy as np


def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)


def caculate_miou(predict, gt, num_class):

    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    for num_batch in range(predict.shape[0]):
        predict_temp = predict[num_batch]
        gt_temp = gt[num_batch]

        acc, pix = accuracy(predict_temp, gt_temp)
        intersection, union = intersectionAndUnion(predict_temp, gt_temp, num_class)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        # print('class [{}], IoU: {}'.format(i, _iou))
        if i == 1:
            iou_fore = _iou

    print('Mean IoU: {:.4}, Accuracy: {:.2f}%'
          .format(iou.mean(), acc_meter.average() * 100))
    return iou.mean(), iou_fore


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap


index2color = labelcolormap(21)
index2color = [list(hh) for hh in index2color]


class Cal_mIoU():
    def __init__(self):
        # super(Cal_mIoU, self).__init__()
        self.acc_meter = AverageMeter()
        self.intersection_meter = AverageMeter()
        self.union_meter = AverageMeter()
        self.iou_each = AverageMeter()

    def caculate_miou(self, predict, gt, num_class):
        for num_batch in range(len(predict)):
            predict_temp = predict[num_batch]
            gt_temp = gt[num_batch]

            acc, pix = accuracy(predict_temp, gt_temp)
            intersection, union = intersectionAndUnion(predict_temp, gt_temp, num_class)
            self.acc_meter.update(acc, pix)
            self.intersection_meter.update(intersection)
            self.union_meter.update(union)
        iou = self.intersection_meter.sum / (self.union_meter.sum + 1e-30)
        for i, _iou in enumerate(iou):
            # print('class [{}], IoU: {}'.format(i, _iou))
            if i == 1:
                iou_fore = _iou

        # print('Mean IoU: {:.4}, Accuracy: {:.2f}%'
        #       .format(iou[1].mean(), self.acc_meter.average() * 100))
        # print('Mean IoU_each', self.iou_each.avg[1])
        return iou.mean(), iou_fore

