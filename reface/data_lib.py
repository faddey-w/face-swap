import os
import copy
import csv
from PIL import Image
import numpy as np
import random
import torch.utils.data
import torch.nn
from collections import defaultdict
from reface.config import Config
from reface.env import device


class Dataset:
    def __init__(self, split_name):
        self._split = split_name
        self._bpath = f".data/bb_landmark/loose_bb_{split_name}.csv"
        self._images_path = f".data/{split_name}"

        with open(self._bpath) as f:
            rdr = csv.reader(f)
            next(rdr)  # header
            self._data = []
            for id_, x, y, w, h in rdr:
                x, y, w, h = map(int, (x, y, w, h))
                self._data.append(
                    {
                        "id": id_,
                        "box": [x, y, x + w, y + h],
                        "file": os.path.join(self._images_path, id_ + ".jpg"),
                    }
                )

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def __iter__(self):
        return iter(self._data)


def get_array(dataset, field):
    return np.array([entry[field] for entry in dataset])


def load_data(entry):
    if "image" not in entry:
        image_pil = Image.open(entry["file"])
        entry["image_size"] = image_pil.size
        entry["image"] = np.array(image_pil)
    return entry


def load_image_size(entry):
    if "image_size" not in entry:
        entry["image_size"] = Image.open(entry["file"]).size
    return entry


class MappedDataset:
    def __init__(self, dataset, cfg, device):
        self.cfg = cfg
        self.dataset = dataset
        self.input_layer = InputLayer(cfg, device)

    def __getitem__(self, item):
        assert isinstance(item, int)
        entry = self.dataset[item]
        entry = copy.deepcopy(entry)
        load_data(entry)
        full_image = entry["image"]

        box = preprocess_face_box(entry["box"], entry["image_size"], self.cfg)

        cropped_image = full_image[box[1] : box[3], box[0] : box[2]]

        entry["image"] = torch.tensor(np.require(cropped_image, requirements="C"))
        entry["image"] = self.input_layer(entry["image"])

        return entry

    def __len__(self):
        return len(self.dataset)


class TrainSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, cfg: Config):
        super().__init__(dataset)
        self.rng = random.Random(cfg.RANDOM_SEED)
        self.batch_size = cfg.INPUT.TRAIN.BATCH_SIZE
        self.n_same_person_pairs = cfg.INPUT.TRAIN.SAME_PERSON_PAIRS_PER_BATCH
        assert self.n_same_person_pairs <= self.batch_size
        self.state_counter = 0
        indices_per_person = defaultdict(list)
        for i, entry in enumerate(dataset):
            pers_id = entry["id"].partition("/")[0]
            indices_per_person[pers_id].append(i)
        self.indices_per_person = dict(indices_per_person)
        self.persons = list(self.indices_per_person.keys())
        self.size = len(dataset)

    def __iter__(self):
        while True:
            if self.state_counter < self.n_same_person_pairs:
                pers_id = self.rng.choice(self.persons)
                i1, i2 = self.rng.choices(self.indices_per_person[pers_id], k=2)
            else:
                pers1_id, pers2_id = self.rng.sample(self.persons, 2)
                i1 = self.rng.choice(self.indices_per_person[pers1_id])
                i2 = self.rng.choice(self.indices_per_person[pers2_id])
            yield i1
            yield i2
            self.state_counter += 1
            self.state_counter %= self.batch_size


def build_data_loader(dataset, cfg: Config, for_training: bool):
    plain_dataset = dataset
    dataset = MappedDataset(dataset, cfg, device)

    if for_training:
        sampler = TrainSampler(plain_dataset, cfg)
        batch_size = cfg.INPUT.TRAIN.BATCH_SIZE
    else:
        sampler = torch.utils.data.SequentialSampler(plain_dataset)
        batch_size = 1
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler,
        # two persons is an item:
        batch_size=2 * batch_size,
        drop_last=True,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.INPUT.LOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=lambda batch: [
            (batch[i], batch[i + 1]) for i in range(0, len(batch), 2)
        ],
    )
    return data_loader


def preprocess_face_box(box, image_size, cfg: Config):
    # extend the face bounding box to give it more context around
    ext_factor = cfg.INPUT.FACE_BOX_EXTENSION_FACTOR
    cx, cy, w, h = box_xyxy_to_cxywh(box)
    box = cx, cy, int(ext_factor * w), int(ext_factor * h)
    box = box_cxywh_to_xyxy(box)

    # box was extended - so now we clip it to image bounds
    image_bounds = (0, 0, *image_size)
    box = box_intersection(box, image_bounds)

    # make it square so we will not need to deal with aspect ratio and so on
    box = box_xyxy_to_cxywh(box)
    sz = min(box[2], box[3])
    box = box_cxywh_to_xyxy([box[0], box[1], sz, sz])

    return box


class InputLayer(torch.nn.Module):
    def __init__(self, cfg: Config, device):
        super(InputLayer, self).__init__()
        self.cfg = cfg
        self.device = device

    def forward(self, image):
        image = image.to(device=self.device)
        image = image.permute([2, 0, 1])  # HWC->CHW
        image = image.to(dtype=torch.float32)

        # TODO: consider normalizing it to unit-normal
        image = image / 255.

        image = image.unsqueeze(0)
        image = torch.nn.functional.interpolate(
            image,
            (self.cfg.INPUT.IMAGE_SIZE, self.cfg.INPUT.IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        image = image.squeeze(0)

        return image


def box_xyxy_to_cxywh(box_xyxy):
    left, top, right, bottom = box_xyxy
    width = right - left
    height = bottom - top
    center_x = left + width // 2
    center_y = top + height // 2
    return center_x, center_y, width, height


def box_cxywh_to_xyxy(box_cxywh):
    center_x, center_y, width, height = box_cxywh
    width_half = width // 2
    height_half = height // 2
    return (
        center_x - width_half,
        center_y - height_half,
        center_x + width_half,
        center_y + height_half,
    )


def box_intersection(box1_xyxy, box2_xyxy):
    return (
        max(box1_xyxy[0], box2_xyxy[0]),
        max(box1_xyxy[1], box2_xyxy[1]),
        min(box1_xyxy[2], box2_xyxy[2]),
        min(box1_xyxy[3], box2_xyxy[3]),
    )