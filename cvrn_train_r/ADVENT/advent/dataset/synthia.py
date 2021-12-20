import numpy as np

from advent.dataset.base_dataset import BaseDataset


class SYNTHIADataSet(BaseDataset):
    def __init__(self, root, list_path, set='all',
                 max_iters=None, crop_size=(321, 321), mean=(128, 128, 128)):
        super().__init__(root, list_path, set, max_iters, crop_size, None, mean)

        # map to cityscape's ids
        # self.id_to_trainid = {3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5,
        #                       15: 6, 9: 7, 6: 8, 16: 9, 1: 10, 10: 11, 17: 12,
        #                       8: 13, 18: 14, 19: 15, 20: 16, 12: 17, 11: 18}
        self.id_to_trainid = {
            3: 0,
            4: 1,
            2: 2,
            21: 3,
            5: 4,
            7: 5,
            15: 6,
            9: 7,
            6: 8,
            1: 9,
            10: 10,
            17: 11,
            8: 12,
            19: 13,
            12: 14,
            11: 15,
        }

    def get_metadata(self, name):
        img_file = self.root / 'RGB' / name
        # label_file = self.root / 'GT/LABELS' / name
        label_file = self.root / 'parsed_LABELS' / name
        return img_file, label_file

    def __getitem__(self, index):
        img_file, label_file, name = self.files[index]
        image = self.get_image(img_file)
        label = self.get_labels(label_file)
        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        image = self.preprocess(image)
        return image.copy(), label_copy.copy(), np.array(image.shape), name
