# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 06:28:10 2022

@author: Jinliang
"""
import json


def get_para(data):
    para_dic = {} # get all the paragraphs: id-text
    for sec in data:
        for para in sec['paragraphs']:
            para_dic[para['paragraph_id']] = para['context']
    return para_dic


def write_json(pgt_dic, path):
    fo = open(path, "w", encoding='utf8')
    count = 0
    for item in pgt_dic.items():
        key = item[0]
        value = item[1]
        line = '{"id": "' + key + '", "text": "' + value + '"}'
        print(count, key)
        fo.write(line + '\n')
        count += 1
    fo.close()
    print('Saving paragraph text to', path)


path = 'data/PGT/data_token.json'
with open(path) as f:
    data = json.load(f)

para = get_para(data)
para_path = "output/retriever/pgt_handbook_paragraph.txt"
write_json(para, para_path)
