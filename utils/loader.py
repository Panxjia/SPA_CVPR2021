# from .transforms import transforms
from torchvision import transforms
from torch.utils.data import DataLoader
from .mydataset import dataset as DataSet
import torch
import imgaug.augmenters as iaa
import numpy as np
import random
import os
import glob

def data_loader(args, test_path=False, train=True):

    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]

    input_size = (int(args.input_size), int(args.input_size))
    crop_size = (int(args.crop_size), int(args.crop_size))


    # transformation for training set
    tsfm_train = transforms.Compose([transforms.Resize(input_size),  # 256
                                     transforms.RandomCrop(crop_size),  # 224
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)])


    # transformation for test cls set
    if args.tencrop == 'True':
        func_transforms = [transforms.Resize(input_size),
                           transforms.TenCrop(crop_size),
                           transforms.Lambda(
                               lambda crops: torch.stack(
                                   [transforms.Normalize(mean_vals, std_vals)(transforms.ToTensor()(crop)) for crop in crops])),
                           ]
    else:
        func_transforms = [transforms.Resize(crop_size),
                           transforms.CenterCrop(crop_size),
                           transforms.ToTensor(),
                           transforms.Normalize(mean_vals, std_vals),]
    tsfm_clstest = transforms.Compose(func_transforms)

    # transformation for test loc set
    tsfm_loctest = transforms.Compose([transforms.Resize(crop_size),  # 224
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)])

    GLOBAL_WORKER_ID = None
    def _init_fn(worker_id):
        global GLOBAL_WORKER_ID
        GLOBAL_WORKER_ID = worker_id
        os.environ['PYTHONHASHSEED'] = str(args.seed + worker_id)
        random.seed(10 + worker_id)
        np.random.seed(10 + worker_id)
        torch.manual_seed(10 + worker_id)
        torch.cuda.manual_seed(10 + worker_id)
        torch.cuda.manual_seed_all(10 + worker_id)

    # training and test dataset & dataloader
    if train:
        img_train = DataSet(args.train_list, root_dir=args.img_dir, transform=tsfm_train, with_path=True,
                                   num_classes=args.num_classes, dataset= args.dataset)
        train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  worker_init_fn=_init_fn)
        return train_loader
    else:
        img_clstest = DataSet(args.test_list, root_dir=args.img_dir, transform=tsfm_clstest, with_path=test_path,
                              num_classes=args.num_classes, dataset=args.dataset)
        img_loctest = DataSet(args.test_list, root_dir=args.img_dir, transform=tsfm_loctest, with_path=test_path,
                              num_classes=args.num_classes, dataset=args.dataset)
        valcls_loader = DataLoader(img_clstest, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        valloc_loader = DataLoader(img_loctest, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        return valcls_loader, valloc_loader

def data_loader_imagenet(args, test_path=False):

    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]

    input_size = (int(args.input_size), int(args.input_size))
    crop_size = (int(args.crop_size), int(args.crop_size))

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(crop_size, scale=(0.08, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_vals, std=std_vals),
    ])

    train_records = glob.glob(args.data_dir + '/train_*.tfrecord')
    train_dataset = ImageTFRecordDataSet(train_records, transform)

    GLOBAL_WORKER_ID = None
    def _init_fn(worker_id):
        global GLOBAL_WORKER_ID
        GLOBAL_WORKER_ID = worker_id
        os.environ['PYTHONHASHSEED'] = str(args.seed + worker_id)
        random.seed(10 + worker_id)
        np.random.seed(10 + worker_id)
        torch.manual_seed(10 + worker_id)
        torch.cuda.manual_seed(10 + worker_id)
        torch.cuda.manual_seed_all(10 + worker_id)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, worker_init_fn=_init_fn)


    # transformation for test cls set
    if args.tencrop == 'True':
        func_transforms = [transforms.Resize(input_size),
                           transforms.TenCrop(crop_size),
                           transforms.Lambda(
                               lambda crops: torch.stack(
                                   [transforms.Normalize(mean_vals, std_vals)(transforms.ToTensor()(crop)) for crop in crops])),
                           ]
    else:
        func_transforms = [transforms.Resize(crop_size),
                           transforms.CenterCrop(crop_size),
                           transforms.ToTensor(),
                           transforms.Normalize(mean_vals, std_vals),]
    tsfm_clstest = transforms.Compose(func_transforms)

    # transformation for test loc set
    tsfm_loctest = transforms.Compose([transforms.Resize(crop_size),  # 224
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)])

    val_records = glob.glob(args.data_dir + '/val_*.tfrecord')
    if len(val_records) == 0:
        print("no val_records found in:{}".format(args.data))

    clsval_dataset = ImageTFRecordDataSet(val_records, tsfm_clstest)
    valcls_loader = DataLoader(clsval_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers,)

    locval_dataset = ImageTFRecordDataSet(val_records, tsfm_loctest)

    valloc_loader = DataLoader(locval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    return train_loader, valcls_loader, valloc_loader



