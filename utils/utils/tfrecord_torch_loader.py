import sys
import io
import os
import struct
import google.protobuf
import time 
import numpy as np
import logging

from threading import local 

from . import yt_example_pb2
from . import yt_feature_pb2

from io import BytesIO
from PIL import Image
import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def tfrecord2idx(tfrecord, idxfile):
    '''
    refer :  https://github.com/NVIDIA/DALI/blob/master/tools/tfrecord2idx
    '''
    if os.access(idxfile, os.R_OK):
        return idxfile
    samples  = 0
    with open(tfrecord, 'rb') as f:
        with open(idxfile, 'w') as idx :
            while True:
                current = f.tell()
                byte_len_crc = f.read(12)
                # eof 
                if len(byte_len_crc) == 0:
                    break
                if len(byte_len_crc) != 12:
                    logging.error("read byte_len_crc failed, file:%s, num:%d pos:%s byte_len_crc:%s" % (tfrecord, samples, f.tell(), len(byte_len_crc)))
                    break
                proto_len = struct.unpack('L', byte_len_crc[:8])[0]
                buffer = f.read(proto_len + 4)
                if len(buffer) != proto_len + 4:
                    logging.error("read proto_len failed, file:%s, num:%d pos:%s proto_len:%s" % (tfrecord, samples, f.tell(), proto_len))
                    break                
                idx.write(str(current) + ' ' + str(f.tell() - current) + '\n')
                samples += 1
    if samples == 0:
        logging.error("no idx found,  file:%s" % tfrecord)
        os.remove(idxfile)
        return None
    logging.info("idx generate done, samples:%s file:%s" %(samples, idxfile))
    return idxfile

class TFRecordDataSet(Dataset):
    def __init__(self, tfrecords):

        tfindexs = [tfrecord2idx(f, f.replace('.tfrecord', '.idx')) for f in tfrecords]
        self.idxs = []
        self.thread_local = local()
        self.thread_local.cache = {}
        self.samples = 0
        for index, tffile in zip(tfindexs, tfrecords):
            idx = []
            with open(index) as idxf:
                for line in idxf:
                    offset, _ = line.split(' ')
                    idx.append(offset)
            self.samples += len(idx)
            print("load %s, samples:%s" %(tffile,  len(idx)))
            self.idxs.append((idx, tffile))


    def __len__(self):
        return self.samples
    
    def parser(self, feature_list):
        raise NotImplementedError("Must Implement parser")
    
    def get_record(self, f, offset):
        f.seek(offset)

        # length,crc
        byte_len_crc = f.read(12)
        proto_len = struct.unpack('Q', byte_len_crc[:8])[0]
        # proto,crc
        pb_data = f.read(proto_len)
        if len(pb_data) < proto_len:
            print("read pb_data err,proto_len:%s pb_data len:%s"%(proto_len, len(pb_data)))
            return None
        
        example = yt_example_pb2.Example()
        example.ParseFromString(pb_data)
        #keep key value in order
        feature = sorted(example.features.feature.items())
     
        record = self.parser(feature)
        #print(record)
        return tuple(record)

    def __getitem__(self, index):
        for idx, tffile in self.idxs:
            if index >= len(idx):
                index -= len(idx)
                continue
            # every thread keep a f instace 
            f = self.thread_local.cache.get(tffile, None)
            if f is None:
                f = open(tffile, 'rb')
                self.thread_local.cache[tffile] = f

            offset = int(idx[index])
            return  self.get_record(f, offset)

        print("bad index,", index)

class ImageTFRecordDataSet(TFRecordDataSet):
    def __init__(self, tfrecords, transforms):
        super(ImageTFRecordDataSet, self).__init__(tfrecords)
        self.transforms = transforms
    def parser(self, feature_list):
        '''
        feature_list = [(key, feature), (key, feature)]
        key is your label.txt col name
        feature is oneof bytes_list, int64_list, float_list
        '''
        for key, feature in feature_list:

            #for image file col
            if key == 'image':
                image_raw = feature.bytes_list.value[0]
                image = Image.open(BytesIO(image_raw))
                image = image.convert('RGB')
                image = self.transforms(image)

            #for int col
            if key == 'label':
                label = feature.int64_list.value[0]
            #for float col
            #if key == 'float_col':
            #   value = feature.float_list.value[0]
            
            # for other str cols
            # if key == 'label_file':
            #    contont = feature.bytes_list.value[0]
            #    #paser the content to value
        return image, label

if __name__ == '__main__':
    from io import BytesIO
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224, scale=(0.08, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    num_shards = 2
    shard_id   = 0
    batch_size = 2
    tfrecords = ['/home/richard/data/imagenet/tfrecord/val_0.tfrecord', '/home/richard/data/imagenet/tfrecord/train_0.tfrecord', '/home/richard/data/imagenet/tfrecord/train_1.tfrecord']

    dataset = ImageTFRecordDataSet(tfrecords, transform)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_shards, rank=shard_id, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)

   
    print("batch_num:%s"% len(dataloader))
    
    end = time.time()
    for i, data in enumerate(dataloader):
        #print(data, type(data))
        image, label = data
        image = image.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        data_time = time.time() - end
        
        print("{} datatime: {:.2f}".format(i, data_time))
        print(i, label, type(label), label.device)
        print(i, image.shape, type(image), image.device)
        end = time.time()
        if i > 2:
            break
print("OK")


