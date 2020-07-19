from reface.faceshifter.AEI_Net import AEI_Net
from reface.faceshifter.MultiscaleDiscriminator import MultiscaleDiscriminator
import torch.optim as optim
from reface.face_recognizer import FaceRecognizer
import torch.nn.functional as F
import torch
import time
import torchvision
import cv2
import os
import itertools
import threading
from queue import Queue, Empty
from typing import Optional
import datetime
from collections import defaultdict
from dataclasses import dataclass
from reface import env, data_lib, utils
from reface.config import Config
from torch.utils.tensorboard import SummaryWriter


class ModelManager:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.cfg = utils.load_config_from_yaml_file(self._get_config_path(model_dir))

    def build_model(self):
        generator = AEI_Net(c_id=512).to(env.device)
        discriminator = MultiscaleDiscriminator(
            input_nc=3, n_layers=6, norm_layer=torch.nn.InstanceNorm2d
        ).to(env.device)
        return generator, discriminator

    def build_optimizer(self, generator, discriminator):
        opt_g = optim.Adam(
            generator.parameters(), lr=self.cfg.MODEL.LR_G, betas=(0, 0.999)
        )
        opt_d = optim.Adam(
            discriminator.parameters(), lr=self.cfg.MODEL.LR_D, betas=(0, 0.999)
        )
        return opt_g, opt_d

    def init_model(self, generator, discriminator):
        for model in [generator, discriminator]:
            for layer in model.modules():
                if isinstance(layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                    torch.nn.init.xavier_normal_(layer.weight)
                    torch.nn.init.constant_(layer.bias, 0)
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.normal_(layer.weight, 0, 0.001)
                    torch.nn.init.constant_(layer.bias, 0)

    def has_checkpoint(self) -> bool:
        return len(self._list_checkpoint_paths()) > 0

    def save_checkpoint(self, generator, discriminator, optimizer_g, optimizer_d, it):
        ckpt = {
            "generator": generator,
            "discriminator": discriminator,
            "optimizer_g": optimizer_g,
            "optimizer_d": optimizer_d,
        }
        ckpt_path = os.path.join(self.model_dir, f"model_{it:07d}.ckpt")
        torch.save(ckpt, ckpt_path)

    def load_from_checkpoint(
        self, generator, discriminator, optimizer_g, optimizer_d, it=None
    ):
        if it is None:
            it, ckpt_path = self._list_checkpoint_paths()[-1]
        else:
            ckpt_path = next(p for i, p in self._list_checkpoint_paths() if i == it)
        ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
        generator.load_state_dict(ckpt["generator"], strict=False)
        discriminator.load_state_dict(ckpt["discriminator"], strict=False)
        optimizer_g.load_state_dict(ckpt["optimizer_g"], strict=False)
        optimizer_d.load_state_dict(ckpt["optimizer_d"], strict=False)
        return it

    @classmethod
    def create_model_dir(cls, cfg, model_dir, strict=True):
        os.makedirs(model_dir, exist_ok=not strict)
        utils.dump_config_to_yaml(cfg, cls._get_config_path(model_dir))

    @staticmethod
    def _get_config_path(model_dir):
        return os.path.join(model_dir, "config.yml")

    def _list_checkpoint_paths(self):
        return [
            (int(name[6:-5]), os.path.join(self.model_dir, name))
            for name in sorted(os.listdir(self.model_dir))
            if name.startswith("model_") and name.endswith(".ckpt")
        ]


class Trainer:
    def __init__(self, model_manager: ModelManager, dataset: data_lib.Dataset):
        self.model_manager = model_manager
        self.cfg = model_manager.cfg
        self.dataset = dataset
        self.face_recognizer = FaceRecognizer()
        self.dataloader = data_lib.build_data_loader(
            self.dataset, self.cfg, for_training=True
        )
        self.data_gen = iter(self.dataloader)

        self.generator, self.discriminator = self.model_manager.build_model()
        self.opt_g, self.opt_d = self.model_manager.build_optimizer(
            self.generator, self.discriminator
        )

        if self.model_manager.has_checkpoint():
            self.start_it = self.model_manager.load_from_checkpoint(
                self.generator, self.discriminator, self.opt_g, self.opt_d
            )
        else:
            self.model_manager.init_model(self.generator, self.discriminator)
            self.start_it = 0
        self.metrics = MetricsAccumulator("2.3f", self.start_it)

        self._summary_writer = SummaryWriter(self.model_manager.model_dir)
        self._last_metrics_q = Queue()

        self._train_thread = None
        self._train_stop_requested = threading.Event()

    def start_train(self):
        self._train_stop_requested.clear()
        self._train_thread = threading.Thread(target=self._train_main)
        self._train_thread.start()

    def stop_train(self, wait=True):
        self._train_stop_requested.set()
        if wait:
            self._train_thread.join()
            self._train_thread = None

    def stream_train_log(self, timeout=180):
        while True:
            it, metrics, metrics_fmt = self._last_metrics_q.get(timeout=timeout)
            now = datetime.datetime.now().time().isoformat()
            message = f"{now} | {it}: " + " ".join(
                f"{key}={valstr}" for key, valstr in metrics_fmt.items()
            )
            vis_it, vis_img = self.metrics.get_last_visualization()
            yield it, metrics, message, vis_it, vis_img

    # noinspection PyTypeChecker,PyUnresolvedReferences
    def _train_main(self):
        batch_size = self.cfg.INPUT.TRAIN.BATCH_SIZE

        self.generator.train()
        self.discriminator.train()

        def hinge_loss(X, positive=True):
            if positive:
                return torch.relu(1 - X).mean()
            else:
                return torch.relu(X + 1).mean()

        for self.it in itertools.count(self.start_it):
            if self._train_stop_requested.isSet():
                break
            step_start_time = time.time()
            data = next(self.data_gen)
            self.metrics.log(data_time=time.time() - step_start_time)

            img_source = data["images1"]
            img_target = data["images2"]
            same_person = data["same_person"]
            with torch.no_grad():
                face_embed_orig = self.face_recognizer(img_source)

            # train G
            self.opt_g.zero_grad()
            img_result, target_attrs = self.generator(img_target, face_embed_orig)

            avd_discr_scores = self.discriminator(img_result)
            loss_g_adv = 0
            for di in avd_discr_scores:
                loss_g_adv += hinge_loss(di[0], True)
            self.metrics.log(loss_g_adv=loss_g_adv.item())

            face_embed_result = self.face_recognizer(img_source)

            loss_g_id = (
                1 - torch.cosine_similarity(face_embed_orig, face_embed_result, dim=1)
            ).mean()
            self.metrics.log(loss_g_id=loss_g_id.item())

            result_attrs = self.generator.get_attr(img_result)
            loss_g_attr = 0
            for i in range(len(target_attrs)):
                loss_g_attr += torch.mean(
                    torch.pow(target_attrs[i] - result_attrs[i], 2).reshape(
                        batch_size, -1
                    ),
                    dim=1,
                ).mean()
            loss_g_attr /= 2.0
            self.metrics.log(loss_g_attr=loss_g_attr.item())

            loss_g_rec = torch.sum(
                0.5
                * torch.mean(
                    torch.pow(img_result - img_target, 2).reshape(batch_size, -1), dim=1
                )
                * same_person
            ) / (same_person.sum() + 1e-6)
            self.metrics.log(loss_g_rec=loss_g_rec.item())

            loss_g = 1 * loss_g_adv + 10 * loss_g_attr + 5 * loss_g_id + 10 * loss_g_rec
            self.metrics.log(loss_g_rec=loss_g_rec.item())
            loss_g.backward()
            self.opt_g.it()

            # train D
            self.opt_d.zero_grad()
            fake_discr_scores = self.discriminator(img_result.detach())
            loss_d_fake = 0
            for di in fake_discr_scores:
                loss_d_fake += hinge_loss(di[0], False)
            self.metrics.log(loss_d_fake=loss_d_fake.item())

            true_discr_scores = self.discriminator(img_source)
            loss_d_true = 0
            for di in true_discr_scores:
                loss_d_true += hinge_loss(di[0], True)
            self.metrics.log(loss_d_true=loss_d_true.item())

            loss_d = 0.5 * (loss_d_true.mean() + loss_d_fake.mean())
            self.metrics.log(loss_d=loss_d.item())

            loss_d.backward()
            self.opt_d.step()

            self.metrics.log(step_time=time.time() - step_start_time)

            if torch.cuda.is_available():
                max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                self.metrics.log(gpu_mem=max_mem_mb, fmt=".0f")

            self.metrics.set_iteration(self.it)

            if (self.it + 1) % self.cfg.LOGGING.VIS_PERIOD_STEP == 0:
                self.metrics.log_visualization(
                    self._make_visualization(img_source, img_target, img_result)
                )
            if (self.it + 1) % self.cfg.LOGGING.PERIOD == 0:
                it, metrics, metrics_formatted = self.metrics.get_values(True)
                for key, val in metrics.items():
                    self._summary_writer.add_scalar(key, val, it)
                try:
                    while True:
                        self._last_metrics_q.get_nowait()
                except Empty:
                    pass
                self._last_metrics_q.put((it, metrics, metrics_formatted))

            if (self.it + 1) % self.cfg.MODEL.CHECKPOINT_PERIOD == 0:
                self.model_manager.save_checkpoint(
                    self.generator, self.discriminator, self.opt_g, self.opt_d, self.it
                )

    def _make_visualization(self, img_source, img_target, img_result):
        def get_grid_image(img):
            img = img[: self.cfg.LOGGING.VIS_MAX_IMAGES]
            img = torchvision.utils.make_grid(img.detach().cpu(), nrow=1)
            return img

        img_source = get_grid_image(img_source)
        img_target = get_grid_image(img_target)
        img_result = get_grid_image(img_result)
        vis_result = torch.cat((img_source, img_result, img_target), dim=2)[::-1, :, :]
        return vis_result.numpy()


class MetricsAccumulator:
    def __init__(self, default_format="s", start_it=0):
        self.default_format = default_format
        self._sums = defaultdict(float)
        self._amounts = defaultdict(int)
        self._formats = {}
        self._lock = threading.Lock()
        self._last_visualization = None
        self._last_visualization_it = None
        self._it = start_it

    def log(self, *, fmt=None, **keys_values):
        with self._lock:
            for key, val in keys_values.items():
                self._sums[key] += val
                self._amounts += 1
                if fmt is not None:
                    self._formats[key] = fmt

    def set_iteration(self, it):
        with self._lock:
            self._it = it

    def log_visualization(self, image):
        with self._lock:
            self._last_visualization = image
            self._last_visualization_it = self._it

    def get_values(self, with_formatted=False, reset=True):
        result = {}
        formatted = {}
        with self._lock:
            for key, val_sum in self._sums.items():
                amount = self._amounts.get(key)
                value = val_sum / amount
                result[key] = value
                if with_formatted:
                    formatted[key] = format(
                        value, self._formats.get(key, self.default_format)
                    )
            if reset:
                self._sums.clear()
                self._amounts.clear()
                self._formats.clear()
            it = self._it
        if with_formatted:
            return it, result, formatted
        else:
            return it, result

    def get_last_visualization(self, reset=True):
        with self._lock:
            image = self._last_visualization
            if reset:
                self._last_visualization = None
            it = self._last_visualization_it

        return it, image
