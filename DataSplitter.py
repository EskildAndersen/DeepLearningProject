'''
Script to seperate labels.txt file into train, dev, and testing files using the
Flickr-8k_'...' txt files only containing image paths. 

The seperated files will now be split into train, dev and testing and contain
both image path and corresponding captions, like labels.txt.
'''

import os
import re

def main():
    lines = readLabels()
    train_labels, dev_labels, test_labels = sortLabels(lines)
    writeLabels(train_labels, 'train')
    writeLabels(dev_labels, 'dev')
    writeLabels(test_labels, 'test')
    

def readLabels():
    file = os.path.join('data', 'texts', 'labels.txt')
    with open(file) as f:
        lines = f.readlines()
        
    return lines


def getImageNames(type: str):
    filename = f'Flickr_8k.{type}Images.txt'
    file = os.path.join('data', 'texts', filename)
    
    with open(file) as f:
        lines = f.readlines()
    
    imageNames = [line.strip() for line in lines]
    
    return imageNames


def sortLabels(lines: list):
    train_names = getImageNames('train')
    dev_names = getImageNames('dev')
    test_names = getImageNames('test')

    train, dev, test = [], [], []
    
    for line in lines:
        img_name, _ = re.split('#\d\t', line)

        counter = 0
        
        if img_name in train_names:
            counter += 1
            train.append(line)
            
        if img_name in dev_names:
            counter += 1
            dev.append(line)
            
        if img_name in test_names:
            counter += 1
            test.append(line)
            
        if counter > 1:
            print('ERROR')
            print(line)
            break
            
    return train, dev, test


def writeLabels(labels: list, type: str):
    filename = f'{type}_labels.txt'
    file = os.path.join('data', 'texts', filename)
    
    with open(file, 'w') as f:
        for label in labels:
            f.write(label)


if __name__ == '__main__':
    main()