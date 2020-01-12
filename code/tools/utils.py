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

