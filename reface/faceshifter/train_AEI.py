from reface.faceshifter.AEI_Net import AEI_Net
from reface.faceshifter.MultiscaleDiscriminator import MultiscaleDiscriminator
import torch.optim as optim
from reface.face_recognizer import FaceRecognizer
import torch
import time
import torchvision
import numpy as np
import os
import tqdm
import logging
import itertools
import threading
import datetime
import visdom
from collections import defaultdict
from reface import env, data_lib, utils, compat
from reface.config import Config
from torch.utils.tensorboard import SummaryWriter


class ModelManager:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.cfg = utils.load_config_from_yaml_file(self._get_config_path(model_dir))
        compat.make_compatible_config(self.cfg)

    # noinspection PyUnresolvedReferences
    def build_model(self):
        generator = AEI_Net(self.cfg).to(env.device)
        discriminator = MultiscaleDiscriminator(self.cfg).to(env.device)
        return generator, discriminator

    def build_optimizer(self, generator, discriminator):
        opt_g = optim.Adam(
            generator.parameters(), lr=self.cfg.GENERATOR.LR, betas=(0, 0.999)
        )
        opt_d = optim.Adam(
            discriminator.parameters(), lr=self.cfg.DISCRIMINATOR.LR, betas=(0, 0.999)
        )
        return opt_g, opt_d

    def init_model(self, generator, discriminator):
        for model in [generator, discriminator]:
            for layer in model.modules():
                if isinstance(layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                    torch.nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.normal_(layer.weight, 0, 0.001)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)

    def has_checkpoint(self) -> bool:
        return len(self._list_checkpoint_paths()) > 0

    def save_checkpoint(self, generator, discriminator, optimizer_g, optimizer_d, it):
        ckpt = {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
        }
        ckpt_path = os.path.join(self.model_dir, f"model_{it:07d}.ckpt")
        torch.save(ckpt, ckpt_path)

        all_ckpts = self._list_checkpoint_paths()
        not_last_ckpts = all_ckpts[: -self.cfg.TRAINING.CHECKPOINTS_MAX_LAST]
        prev_keep_i = 0
        keep_period = self.cfg.TRAINING.CHECKPOINTS_KEEP_PERIOD
        for old_ckpt_i, old_ckpt_path in not_last_ckpts:
            if keep_period is not None:
                if old_ckpt_i >= prev_keep_i + keep_period:
                    prev_keep_i = old_ckpt_i
                    continue
            os.remove(old_ckpt_path)

        return ckpt_path

    def load_from_checkpoint(
        self,
        generator,
        discriminator=None,
        optimizer_g=None,
        optimizer_d=None,
        it=None,
        ckpt_path=None,
    ):
        if ckpt_path is not None:
            assert it is None
        elif it is not None:
            ckpt_path = next(p for i, p in self._list_checkpoint_paths() if i == it)
        else:
            it, ckpt_path = self._list_checkpoint_paths()[-1]
        ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
        generator.load_state_dict(ckpt["generator"], strict=False)
        if discriminator is not None:
            discriminator.load_state_dict(ckpt["discriminator"], strict=False)
        if optimizer_g is not None:
            optimizer_g.load_state_dict(ckpt["optimizer_g"])
        if optimizer_d is not None:
            optimizer_d.load_state_dict(ckpt["optimizer_d"])
        return it

    def list_checkpoints(self):
        return [i for i, p in self._list_checkpoint_paths()]

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
    def __init__(
        self,
        model_manager: ModelManager,
        train_dataset: data_lib.Dataset,
        test_dataset: data_lib.Dataset,
        visdom_addr=None,
    ):
        if visdom_addr is None:
            visdom_addr = "http://0.0.0.0", 8097
        elif isinstance(visdom_addr, str):
            visdom_addr = visdom_addr, 8097
        elif isinstance(visdom_addr, int):
            visdom_addr = "http://0.0.0.0", visdom_addr
        self.model_manager = model_manager
        self.cfg: Config = model_manager.cfg

        if self.cfg.INPUT.MIN_FACE_SIZE is not None:
            min_face_size = self.cfg.INPUT.MIN_FACE_SIZE
            train_dataset = [
                entry
                for entry in tqdm.tqdm(
                    train_dataset,
                    desc=f"train: filtering out faces < {min_face_size}px",
                )
                if get_face_size(entry) >= min_face_size
            ]
            test_dataset = [
                entry
                for entry in tqdm.tqdm(
                    test_dataset, desc=f"test: filtering out faces < {min_face_size}px"
                )
                if get_face_size(entry) >= min_face_size
            ]

        _log.info("train size: %s", len(train_dataset))
        _log.info("test size: %s", len(test_dataset))
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.face_recognizer = FaceRecognizer()
        self.dataloader = data_lib.build_data_loader(
            self.train_dataset, self.cfg, for_training=True
        )
        self.data_gen = iter(self.dataloader)
        self.dataloader_test = data_lib.build_data_loader(
            self.test_dataset,
            self.cfg,
            for_training=False,
            num_workers=max(1, self.cfg.INPUT.LOADER.NUM_WORKERS),
        )
        self.data_gen_test = iter(self.dataloader_test)

        torch.random.manual_seed(self.cfg.RANDOM_SEED)

        self.generator, self.discriminator = self.model_manager.build_model()
        self.opt_g, self.opt_d = self.model_manager.build_optimizer(
            self.generator, self.discriminator
        )

        if self.model_manager.has_checkpoint():
            self.start_it = self.model_manager.load_from_checkpoint(
                self.generator, self.discriminator, self.opt_g, self.opt_d
            )
        else:
            if self.cfg.TRAINING.INIT_CHECKPOINT is None:
                self.model_manager.init_model(self.generator, self.discriminator)
            else:
                self.model_manager.load_from_checkpoint(
                    self.generator,
                    self.discriminator,
                    ckpt_path=self.cfg.TRAINING.INIT_CHECKPOINT,
                )
            self.start_it = 0
        self._visdom_addr = visdom_addr
        self._writers_initialized = False

    def _init_writers(self):
        if self._writers_initialized:
            return
        self._metrics = MetricsAccumulator("2.3f", self.start_it)

        self._summary_writer = SummaryWriter(
            self.model_manager.model_dir, flush_secs=30
        )
        self._log_file = open(
            os.path.join(self.model_manager.model_dir, "log.txt"), "a"
        )
        self._vis = visdom.Visdom(
            self._visdom_addr[0],
            port=self._visdom_addr[1],
            env=os.path.relpath(self.model_manager.model_dir, env.models_dir),
        )
        self._writers_initialized = True

    def train(self):
        self._init_writers()
        self.generator.train()
        self.discriminator.train()

        step_period = self.cfg.TRAINING.OPT_STEP_PERIOD = 1
        loss_coeff = torch.tensor(1.0 / step_period).to(env.device)

        self._metrics.log(data_failures=0)

        for self.it in itertools.count(self.start_it):
            step_start_time = time.time()
            data = next(self.data_gen)

            if data["failed_image_ids"]:
                sampler = self.dataloader.batch_sampler.sampler
                sampler.disable_failed_entries(data["failed_image_ids"])
                self._metrics.log(
                    data_failures=sampler.total_failed_entries
                    / ((self.it + 1) * self.cfg.INPUT.TRAIN.BATCH_SIZE)
                )

            self._metrics.log(time_data=time.time() - step_start_time)

            img_source = data["images1"].to(env.device)
            img_target = data["images2"].to(env.device)
            same_person = data["same_person"].to(env.device)

            # inference
            with torch.no_grad():
                face_embed_orig = self.face_recognizer(img_source)
            img_result, target_attrs = self.generator(img_target, face_embed_orig)

            # train D
            if self.it % step_period == 0:
                self.opt_d.zero_grad()
            losses_d = self._get_discriminator_losses(img_target, img_result.detach())
            self._metrics.log(
                Loss_D_fake=float(losses_d["fake"]),
                Loss_D_true=float(losses_d["true"]),
                Loss_D=float(losses_d["total"]),
            )

            losses_d["total"].backward(loss_coeff)
            if self.it % step_period == 0:
                self.opt_d.step()

            # train G
            if self.it % step_period == 0:
                self.opt_g.zero_grad()
            losses_g = self._get_generator_losses(
                face_embed_orig, img_target, target_attrs, img_result, same_person
            )
            self._metrics.log(
                Loss_G_adv=float(losses_g["adv"]),
                Loss_G_id=float(losses_g["id"]),
                Loss_G_attr=float(losses_g["attr"]),
                Loss_G_rec=float(losses_g["rec"]),
                Loss_G=float(losses_g["total"]),
            )
            for p in self.discriminator.parameters():
                p.requires_grad = False
            losses_g["total"].backward(loss_coeff)
            for p in self.discriminator.parameters():
                p.requires_grad = True
            if self.it % step_period == 0:
                self.opt_g.step()

            self._metrics.log(time_step=time.time() - step_start_time)

            if torch.cuda.is_available():
                max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                self._metrics.log(mem_gpu=max_mem_mb, fmt=".0f")

            self._metrics.set_iteration(self.it)

            if (self.it + 1) % self.cfg.TRAINING.VIS_PERIOD == 0:
                self._visualize_train(img_source, img_target, img_result)
            if (self.it + 1) % self.cfg.TEST.VIS_PERIOD == 0:
                self._visualize_test()
            if (
                self.it != self.start_it
                and self.it % self.cfg.TRAINING.CHECKPOINT_PERIOD == 0
            ):
                ckpt_path = self.model_manager.save_checkpoint(
                    self.generator, self.discriminator, self.opt_g, self.opt_d, self.it
                )
                _log.info("checkpoint: %s", ckpt_path)
            if self.it != self.start_it and self.it % self.cfg.TEST.TEST_PERIOD == 0:
                self._test()
            if self.it != self.start_it and self.it % self.cfg.TRAINING.LOG_PERIOD == 0:
                self._log_metrics()

    def _get_generator_losses(
        self, face_embed, img_target, target_attrs, img_result, same_person
    ):
        batch_size = img_target.shape[0]
        avd_discr_scores = self.discriminator(img_result)
        loss_g_adv = 0
        for d_scores_map in avd_discr_scores:
            loss_g_adv += hinge_loss(d_scores_map, True)
        loss_g_adv *= 0.1

        face_embed_result = self.face_recognizer(img_result)

        # noinspection PyTypeChecker,PyUnresolvedReferences
        loss_g_id = (
            0.5
            * (1 - torch.cosine_similarity(face_embed, face_embed_result, dim=1)).mean()
        )

        result_attrs = self.generator.get_attr(img_result)
        loss_g_attr = 0
        for i in range(len(target_attrs)):
            loss_g_attr += torch.mean(
                torch.pow(target_attrs[i] - result_attrs[i], 2).reshape(batch_size, -1),
                dim=1,
            ).mean()
        loss_g_attr /= 2.0

        loss_g_rec = torch.sum(
            0.5
            * torch.mean(
                torch.pow(img_result - img_target, 2).reshape(batch_size, -1), dim=1
            )
            * same_person
        ) / (same_person.sum() + 1e-6)

        loss_g = loss_g_adv + loss_g_attr + loss_g_id + loss_g_rec

        return dict(
            adv=loss_g_adv, attr=loss_g_attr, id=loss_g_id, rec=loss_g_rec, total=loss_g
        )

    def _get_discriminator_losses(self, img_real, img_generated):
        fake_discr_scores = self.discriminator(img_generated)
        loss_d_fake = 0
        for d_scores_map in fake_discr_scores:
            loss_d_fake += hinge_loss(d_scores_map, False)

        true_discr_scores = self.discriminator(img_real)
        loss_d_true = 0
        for d_scores_map in true_discr_scores:
            loss_d_true += hinge_loss(d_scores_map, True)

        # noinspection PyTypeChecker
        loss_d = 0.5 * (torch.mean(loss_d_true) + torch.mean(loss_d_fake))

        return dict(true=loss_d_true, fake=loss_d_fake, total=loss_d)

    def _test(self):
        self.generator.eval()
        self.discriminator.eval()
        for _ in tqdm.trange(self.cfg.TEST.N_TEST_BATCHES, desc="TEST"):
            data = next(self.data_gen_test)
            img_source = data["images1"].to(env.device)
            img_target = data["images2"].to(env.device)
            same_person = data["same_person"].to(env.device)
            with torch.no_grad():
                face_embed = self.face_recognizer(img_source)
                img_result, target_attrs = self.generator(img_target, face_embed)
                losses_g = self._get_generator_losses(
                    face_embed, img_target, target_attrs, img_result, same_person
                )
                self._metrics.log(
                    LossTest_G_adv=float(losses_g["adv"]),
                    LossTest_G_id=float(losses_g["id"]),
                    LossTest_G_attr=float(losses_g["attr"]),
                    LossTest_G_rec=float(losses_g["rec"]),
                    LossTest_G=float(losses_g["total"]),
                )
                losses_d = self._get_discriminator_losses(img_target, img_result)
                self._metrics.log(
                    LossTest_D_fake=float(losses_d["fake"]),
                    LossTest_D_true=float(losses_d["true"]),
                    LossTest_D=float(losses_d["total"]),
                )
        self.generator.train()
        self.discriminator.train()

    def _visualize_train(self, img_source, img_target, img_result):
        self._vis.image(
            self._make_visualization(
                img_source, img_target, img_result, self.cfg.TRAINING.VIS_MAX_IMAGES
            ),
            "AEI-train",
            opts=dict(caption="AEI-train"),
        )

    def _visualize_test(self):
        self._vis.image(
            self.get_demo("test", self.cfg.TEST.VIS_MAX_IMAGES),
            "AEI-test",
            opts=dict(caption="AEI-test"),
        )

    def _log_metrics(self):
        it, metrics, metrics_formatted = self._metrics.get_values(True)
        for key, val in metrics.items():
            section, _, subkey = key.partition("_")
            key = section + "/" + subkey
            self._summary_writer.add_scalar(key, val, it)
        now = datetime.datetime.now().time().isoformat()
        message = f"{now} | {it}: " + " ".join(
            f"{key}={valstr}" for key, valstr in metrics_formatted.items()
        )
        print(message, file=self._log_file, flush=True)
        _log.info(message)

    def _make_visualization(self, img_source, img_target, img_result, max_images=None):
        def get_grid_image(img):
            if max_images is not None:
                step = img.shape[0] // max_images
                if step > 0:
                    img = img[: step * max_images : step]
            img = torchvision.utils.make_grid(img.detach().cpu(), nrow=1)
            return img

        img_source = get_grid_image(img_source)
        img_target = get_grid_image(img_target)
        img_result = get_grid_image(img_result)
        vis_result = torch.cat((img_source, img_result, img_target), dim=2)
        vis_result = (vis_result + 1) / 2
        return vis_result

    def get_demo(self, which_data, max_images):
        if which_data == "train":
            data = next(self.data_gen)
        elif which_data == "test":
            data = next(self.data_gen_test)
        else:
            raise ValueError(which_data)
        img_source = data["images1"].to(env.device)
        img_target = data["images2"].to(env.device)

        self.generator.eval()
        self.discriminator.eval()
        with torch.no_grad():
            face_embed = self.face_recognizer(img_source)
            img_result, target_attrs = self.generator(img_target, face_embed)

        self.generator.train()
        self.discriminator.train()
        return self._make_visualization(img_source, img_target, img_result, max_images)


class MetricsAccumulator:
    def __init__(self, default_format="s", start_it=0):
        self.default_format = default_format
        self._sums = defaultdict(float)
        self._amounts = defaultdict(int)
        self._formats = {}
        self._lock = threading.Lock()
        self._it = start_it

    def log(self, *, fmt=None, **keys_values):
        with self._lock:
            for key, val in keys_values.items():
                self._sums[key] += val
                self._amounts[key] += 1
                if fmt is not None:
                    self._formats[key] = fmt

    def set_iteration(self, it):
        with self._lock:
            self._it = it

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


def hinge_loss(X, positive=True):
    if positive:
        return torch.relu(1 - X).mean()
    else:
        return torch.relu(X + 1).mean()


def get_face_size(entry):
    return min(data_lib.box_xyxy_to_cxywh(entry["box"])[2:])


_log = logging.getLogger(__name__)
