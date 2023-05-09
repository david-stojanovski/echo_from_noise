class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def add_data2tb(writer, log_data, epoch):
    writer.add_scalar('dice_mean', log_data['dice_mean'], epoch)
    writer.add_scalar('1/dice_lv_endo', log_data['dice_lv_endo'], epoch)
    writer.add_scalar('1/dice_atrium', log_data['dice_atrium'], epoch)
    writer.add_scalar('1/dice_lv_epi', log_data['dice_lv_epi'], epoch)
    return
