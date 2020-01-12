"""
@author: Dongze Lian
@contact: liandz@shanghaitech.edu.cn
@software: PyCharm
@file: utils.py
@time: 2020/1/11 23:42
"""

def txt2list(txt_file = ''):
    txt_list = []
    with open(txt_file, 'r') as file:
        for line in file.readlines():
            txt_list.append(line.strip('\n'))

    return txt_list


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



def infinite_get(data_iter, data_queue):
    try:
        data = next(data_iter)
    except StopIteration:
        # StopIteration is thrown if dataset ends
        # reinitialize data loader
        data_iter = iter(data_queue)
        data = next(data_iter)
    return data, data_iter