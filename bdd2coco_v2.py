import os
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='bdd2coco')
# parser.add_argument('--bdd_dir', type=str, default='E:\\bdd100k')
parser.add_argument('--bdd_dir', type=str, default='data/bdd100k/labels/')
cfg = parser.parse_args()

# src_val_dir = os.path.join(cfg.bdd_dir, 'labels', 'bdd100k_labels_images_val.json')
# src_train_dir = os.path.join(cfg.bdd_dir, 'labels', 'bdd100k_labels_images_train.json')
src_val_dir = os.path.join(cfg.bdd_dir, 'det_20', 'cut_det_val.json')
src_train_dir = os.path.join(cfg.bdd_dir, 'det_20', 'cut_det_train.json')


os.makedirs(os.path.join(cfg.bdd_dir, 'det_20_coco'), exist_ok=True)

dst_val_dir = os.path.join(cfg.bdd_dir, 'det_20_coco', 'cut_det_val_coco_v2.json')
dst_train_dir = os.path.join(cfg.bdd_dir, 'det_20_coco', 'cut_det_train_coco_v2.json')


def bdd2coco_detection(labeled_images, save_dir):

  attr_dict = {"categories":
    [{"supercategory": "none", "id": 1, "name": "pedestrian"},
    {"supercategory": "none", "id": 1, "name": "other person"},
    {"supercategory": "none", "id": 1, "name": "rider"},
    {"supercategory": "none", "id": 2, "name": "bicycle"},
    {"supercategory": "none", "id": 3, "name": "car"},
    {"supercategory": "none", "id": 4, "name": "motorcycle"},
    {"supercategory": "none", "id": 5, "name": "bus"},
     {"supercategory": "none", "id": 6, "name": "train"},
     {"supercategory": "none", "id": 7, "name": "truck"},
     {"supercategory": "none", "id": 8, "name": "traffic light"}]}

  id_dict = {i['name']: i['id'] for i in attr_dict['categories']}

  images = list()
  annotations = list()
  ignore_categories = set()

  counter = 0
  for i in tqdm(labeled_images):

    # for "images"
    counter += 1
    image = dict()
    image['file_name'] = i['name']
    image['height'] = 720
    image['width'] = 1280
    image['id'] = counter

    empty_image = True

    tmp = 0

    try: # Avoid KeyError 
        for l in i['labels']:
          annotation = dict()
          if l['category'] in id_dict.keys(): # if there is at least one class in the image
              tmp = 1
              empty_image = False
              annotation["iscrowd"] = 0
              annotation["image_id"] = image['id']
              annotation["file_name"] = image['file_name']
#               annotation['image_id'] = image['id'] # for identification. for every box, it has its own id
              x1 = l['box2d']['x1']
              y1 = l['box2d']['y1']
              x2 = l['box2d']['x2']
              y2 = l['box2d']['y2']
              annotation['bbox'] = [x1, y1, x2 - x1, y2 - y1]
              # annotation['area'] = float((x2 - x1) * (y2 - y1)) #segmentation area
              annotation['category_id'] = id_dict[l['category']]
              annotation['ignore'] = 0
              annotation['id'] = l['id'] # id in an image
              # annotation['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
              annotations.append(annotation)
          else:
              ignore_categories.add(l['category'])
    except KeyError:
        print("skip no label")
    

    if empty_image:
      print('empty image!')
      continue
    if tmp == 1: # if there is at least one class in the image
      images.append(image)

  attr_dict["images"] = images
  attr_dict["annotations"] = annotations
  # for rename of some classes--> bdd.names
  attr_dict["categories"] = [ \
      {"supercategory": "person", "id": 1, "name": "person"}, \
      {"supercategory": "vehicle", "id": 2, "name": "bicycle"}, \
      {"supercategory": "vehicle", "id": 3, "name": "car"}, \
      {"supercategory": "vehicle", "id": 4, "name": "motorcycle"}, \
      {"supercategory": "vehicle", "id": 5, "name": "bus"}, \
      {"supercategory": "vehicle", "id": 6, "name": "train"},\
      {"supercategory": "vehicle", "id": 7, "name": "truck"}, \
      {"supercategory": "outdoor", "id": 8, "name": "traffic light"}]
  # attr_dict["type"] = "instances"

  print('ignored categories: ', ignore_categories)
  print('saving...')
  with open(save_dir, "w") as file:
    json.dump(attr_dict, file)
  print('Done.')


def main():
  # create BDD training set detections in COCO format
  print('Loading training set...')
  with open(src_train_dir) as f:
    train_labels = json.load(f)
  print('Converting training set...')
  bdd2coco_detection(train_labels, dst_train_dir)

  # create BDD validation set detections in COCO format
  print('Loading validation set...')
  with open(src_val_dir) as f:
    val_labels = json.load(f)
  print('Converting validation set...')
  bdd2coco_detection(val_labels, dst_val_dir)


if __name__ == '__main__':
  main()