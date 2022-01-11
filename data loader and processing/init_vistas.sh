sed -i 's/semantic = np.zeros(pan.shape, dtype=np.uint8)/semantic = np.ones(pan.shape, dtype=np.uint8) * 255/g' lib/dataset_devkit/panopticapi/converters/panoptic2semantic_segmentation.py

PYTHONPATH=$(pwd)/lib/dataset_devkit:$(pwd)/lib/dataset_devkit/panopticapi:$PYTHONPATH python lib/dataset_devkit/panopticapi/converters/panoptic2semantic_segmentation.py \
--input_json_file data/vistas/annotations/panoptic_train/panoptic_2018.json --segmentations_folder data/vistas/annotations/panoptic_train \
--semantic_seg_folder data/vistas/annotations/panoptic_train_semantic_trainid --categories_json_file data/vistas/annotations/panoptic_vistas_categories.json

PYTHONPATH=$(pwd)/lib/dataset_devkit:$(pwd)/lib/dataset_devkit/panopticapi:$PYTHONPATH python lib/dataset_devkit/panopticapi/converters/panoptic2detection_coco_format.py \
--output_json_file data/vistas/annotations/stuff_train.json \
--input_json_file data/vistas/annotations/panoptic_train/panoptic_2018.json --segmentations_folder data/vistas/annotations/panoptic_train \
--categories_json_file data/vistas/annotations/panoptic_vistas_categories.json

