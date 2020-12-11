import matplotlib.pyplot as plt
import numpy as np
import json
import torch
import os
import torchvision
import transforms as T
from engine import train_one_epoch, evaluate
import utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from itertools import groupby
from pycocotools import mask as maskutil

class Pascal_train_dataset(Dataset):
    def __init__(self, root, jsfile, transform):
        self.root = root
        self.transform = transform
        self.coco = COCO(jsfile)
    
    def __len__(self):
        return len(list(self.coco.imgs.keys()))
    
    def __getitem__(self, idx):
        #load image info
        img_id = list(self.coco.imgs.keys())[idx]
        img_info = self.coco.loadImgs(ids=img_id)

        #open image
        img_path = self.root + img_info[0]['file_name']
        image = Image.open(img_path)

        #load instance
        annids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(annids)

        #create fields
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        masks  = []
        for i in range(len(annids)):
            x1 = anns[i]['bbox'][0] + anns[i]['bbox'][2]
            y1 = anns[i]['bbox'][1] + anns[i]['bbox'][3]
            bbox = [anns[i]['bbox'][0], anns[i]['bbox'][1], x1, y1]
            boxes.append(bbox)
            labels.append(anns[i]['category_id'])
            areas.append(anns[i]['area'])
            iscrowd.append(anns[i]['iscrowd'])
            masks.append(self.coco.annToMask(anns[i]))

        #convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype = torch.float32) 
        labels = torch.as_tensor(labels, dtype = torch.int64)
        img_id = torch.tensor([img_id], dtype = torch.int64)
        areas = torch.as_tensor(areas)
        iscrowd = torch.as_tensor(iscrowd, dtype = torch.uint8)
        masks = torch.as_tensor(masks)
        

        #build target dict
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id
        target["area"] = areas
        target["iscrowd"] = iscrowd
        target["masks"] = masks

        #transform image and bbox
        if self.transform is not None:
                image, target = self.transform(image, target)

        return image, target
    
class Pascal_test_dataset(Dataset):
    def __init__(self, root, jsfile):
        self.root = root
        self.transform = transforms.ToTensor()
        self.coco = COCO(jsfile)

    def __len__(self):
        return len(list(self.coco.imgs.keys()))

    def __getitem__(self, idx):
        #load image info
        img_id = list(self.coco.imgs.keys())[idx]
        img_info = self.coco.loadImgs(ids=img_id)

        #open image
        img_path = self.root + img_info[0]['file_name']
        image = Image.open(img_path)

        #transform image
        if self.transform is not None:
                image = self.transform(image)

        return image

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on Imagenet
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained_backbone=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    compressed_rle = maskutil.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
    compressed_rle['counts'] = str(compressed_rle['counts'], encoding='utf-8')
    return compressed_rle

def main(train):
    os.environ["CUDA_VISIBLE_DEVICES"]="7"
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 21 #20 class + 1

    print('loading')
    # use our dataset and defined transformations
    train_dataset = Pascal_train_dataset('train_images/', 'pascal_train.json', get_transform(True))
    test_dataset = Pascal_test_dataset('test_images/', 'test.json')

    # define training and validation data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    fname_header = 'pascal_detection_'
    fname_tail = '.pth'
    if train:
        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        print('training')
        # let's train it for 42 epochs
        num_epochs = 42

        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()

            if (epoch + 1) % 10 == 0:
                #save model
                PATH = fname_header + str(epoch + 1) + fname_tail
                torch.save(model.state_dict(), PATH)

        #save model
        PATH = fname_header + 'final' + fname_tail
        torch.save(model.state_dict(), PATH)
    else:
        PATH = fname_header + fname_tail
        model.load_state_dict(torch.load(PATH))
        print("load " + PATH)
    
    print("testing")
    ans = []
    with torch.no_grad():
        # For inference
        model.eval()
        for idx, img in enumerate(test_dataset):
            #print(img.shape)
            img = img.to(device)

            predictions = model(img[None, ...])
            predictions = [{k: v.to("cpu") for k, v in p.items()} for p in predictions]
            #print(predictions) #boxes, labels, scores, masks
            
            n_instances = len(predictions[0]['scores'])
            for i in range(n_instances):
                pred = {}
                pred['image_id'] = list(test_dataset.coco.imgs.keys())[idx]
                pred['category_id'] = int(predictions[0]['labels'][i])
                masks = np.where(predictions[0]['masks'][i].numpy() > 0.4, 1, 0).astype(np.uint8)[0, :, :]
                pred['segmentation'] = binary_mask_to_rle(masks)
                pred['score'] = float(predictions[0]['scores'][i])
                ans.append(pred)
            if (idx + 1) % 10 == 0:
                print(idx)

        with open("309551123.json", "w") as f:
            json.dump(ans, f)
    

if __name__ == "__main__":
    train = True
    main(train)
