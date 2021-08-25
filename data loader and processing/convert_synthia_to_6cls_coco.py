from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
# import h5py
import json
import os
import scipy.misc
import sys
import numpy as np

import cityscapesscripts.evaluation.instances2dict_with_polygons as cs


def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument(
        '--dataset', default='synthia', type=str)
    parser.add_argument(
        '--outdir', help="output dir for json files", default='./datasets/synthia/annotations', type=str)
    parser.add_argument(
        '--datadir', help="data dir for annotations to be converted", default='/data/obj_det/da/datasets/synthia', type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')


def polys_to_boxes(polys):
    """Convert a list of polygons into an array of tight bounding boxes."""
    boxes_from_polys = np.zeros((len(polys), 4), dtype=np.float32)
    for i in range(len(polys)):
        poly = polys[i]
        x0 = min(min(p[::2]) for p in poly)
        x1 = max(max(p[::2]) for p in poly)
        y0 = min(min(p[1::2]) for p in poly)
        y1 = max(max(p[1::2]) for p in poly)
        boxes_from_polys[i, :] = [x0, y0, x1, y1]

    return boxes_from_polys

def convert_synthia_instance_only(
        data_dir, out_dir):
    """Convert from synthia format to COCO instance seg format - polygons"""
    sets = [
        'images',
    ]
    ann_dirs = [
        'labels',
    ]
    json_name = '6cls_filtered_train.json'
    ends_in = 'InstanceIds.png'
    img_id = 0
    ann_id = 0
    cat_id = 1
    category_dict = {}

    category_instancesonly = [
        'person',
        'rider',
        'car',
        'bus',
        'motorcycle',
        'bicycle',
    ]

    for cat in category_instancesonly:
        category_dict[cat] = cat_id
        cat_id += 1

    for data_set, ann_dir in zip(sets, ann_dirs):
        print('Starting %s' % data_set)
        ann_dict = {}
        images = []
        annotations = []
        ann_dir = os.path.join(data_dir, ann_dir)
        for root, _, files in os.walk(ann_dir):
            files.sort()
            for filename in files:
                if filename.endswith(ends_in):
                    if len(images) % 50 == 0:
                        print("Processed %s images, %s annotations" % (
                            len(images), len(annotations)))

                    # json_ann = json.load(open(os.path.join(root, filename)))

                    image = {}
                    image['id'] = img_id
                    img_id += 1
                    image['width'] = 1280
                    image['height'] = 760

                    image['file_name'] = filename.replace('_InstanceIds.png','.png')
                    image['seg_file_name'] = filename
                    images.append(image)

                    fullname = os.path.join(root, image['seg_file_name'])
                    objects = cs.instances2dict_with_polygons(
                        [fullname], verbose=False)[fullname]

                    # import pdb
                    # pdb.set_trace()

                    for object_cls in objects:

                        if object_cls not in category_instancesonly:
                            continue  # skip non-instance categories

                        for obj in objects[object_cls]:

                            if obj['contours'] == []:
                                print('Warning: empty contours.')
                                continue  # skip non-instance categories

                            # len_p = [len(p) for p in obj['contours']]
                            # if min(len_p) <= 4:
                            #     print('Warning: invalid contours.')
                            #     continue  # skip non-instance categories

                            if obj['pixelCount'] <= 64:
                                # print('Warning: skip too small area.')
                                continue

                            contours = []
                            for contour in obj['contours']:
                                # skip too small area
                                if len(contour) > 4:
                                    contours.append(contour)
                            if not contours:
                                continue

                            ann = {}
                            ann['id'] = ann_id
                            ann_id += 1
                            ann['image_id'] = image['id']
                            # ann['segmentation'] = obj['contours']
                            ann['segmentation'] = contours

                            if object_cls not in category_dict:
                                category_dict[object_cls] = cat_id
                                cat_id += 1
                            ann['category_id'] = category_dict[object_cls]
                            ann['iscrowd'] = 0
                            ann['area'] = obj['pixelCount']
                            ann['bbox'] = xyxy_to_xywh(
                                polys_to_boxes(
                                    [ann['segmentation']])).tolist()[0]

                            annotations.append(ann)

                # import pdb
                # pdb.set_trace()

        ann_dict['images'] = images
        categories = [{"id": category_dict[name], "name": name} for name in
                      category_instancesonly]
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations
        print("Num categories: %s" % len(categories))
        print(categories)
        print("Num images: %s" % len(images))
        print("Num annotations: %s" % len(annotations))
        with open(os.path.join(out_dir, json_name), 'w') as outfile:
            outfile.write(json.dumps(ann_dict))


if __name__ == '__main__':
    args = parse_args()
    if args.dataset == "synthia":
        convert_synthia_instance_only(args.datadir, args.outdir)
    else:
        print("Dataset not supported: %s" % args.dataset)
