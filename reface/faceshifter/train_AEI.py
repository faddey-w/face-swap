from reface.faceshifter.AEI_Net import AEI_Net
from reface.faceshifter.MultiscaleDiscriminator import MultiscaleDiscriminator
from utils.Dataset import FaceEmbed
from torch.utils.data import DataLoader
import torch.optim as optim
from reface.face_recognizer import get_face_recognizer
import torch.nn.functional as F
import torch
import time
import torchvision
import cv2
import os
from reface import env, data_lib, utils
from reface.config import Config


class ModelManager:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.cfg = utils.load_config_from_yaml_file(self._get_config_path(model_dir))
        self.generator = AEI_Net(c_id=512).to(env.device)
        self.discriminator = MultiscaleDiscriminator(
            input_nc=3, n_layers=6, norm_layer=torch.nn.InstanceNorm2d
        ).to(env.device)

    @classmethod
    def init_model(cls, cfg, model_dir, strict=True):
        os.makedirs(model_dir, exist_ok=not strict)
        utils.dump_config_to_yaml(cfg, cls._get_config_path(model_dir))

    @staticmethod
    def _get_config_path(model_dir):
        return os.path.join(model_dir, "config.yml")


class Trainer:
    def __init__(self, cfg: Config, dataset: data_lib.Dataset):
        self.cfg = cfg
        self.dataset = dataset

        self.face_recognizer = get_face_recognizer()

    def _train_main(self):
        batch_size = self.cfg.INPUT.TRAIN.BATCH_SIZE
        max_epoch = 2000
        show_step = 10
        optim_level = self.cfg.MODEL.AMP_LEVEL

        G = self.generator
        D = self.discriminator
        G.train()
        D.train()

        opt_G = optim.Adam(G.parameters(), lr=self.cfg.MODEL.LR_G, betas=(0, 0.999))
        opt_D = optim.Adam(D.parameters(), lr=self.cfg.MODEL.LR_D, betas=(0, 0.999))

        G, opt_G = amp.initialize(G, opt_G, opt_level=optim_level)
        D, opt_D = amp.initialize(D, opt_D, opt_level=optim_level)

        try:
            G.load_state_dict(
                torch.load(
                    "./saved_models/G_latest.pth", map_location=torch.device("cpu")
                ),
                strict=False,
            )
            D.load_state_dict(
                torch.load(
                    "./saved_models/D_latest.pth", map_location=torch.device("cpu")
                ),
                strict=False,
            )
        except Exception as e:
            print(e)

        dataloader = data_lib.build_data_loader(self.dataset, self.cfg, for_training=True)

        def hinge_loss(X, positive=True):
            if positive:
                return torch.relu(1 - X).mean()
            else:
                return torch.relu(X + 1).mean()

        def get_grid_image(X):
            X = X[:8]
            X = (
                torchvision.utils.make_grid(X.detach().cpu(), nrow=X.shape[0]) * 0.5
                + 0.5
            )
            return X

        def make_image(Xs, Xt, Y):
            Xs = get_grid_image(Xs)
            Xt = get_grid_image(Xt)
            Y = get_grid_image(Y)
            return torch.cat((Xs, Xt, Y), dim=1).numpy()

        # prior = torch.FloatTensor(cv2.imread('./prior.png', 0).astype(np.float)/255).to(device)

        for iteration, data in enumerate(dataloader):
            start_time = time.time()
            Xs, Xt, same_person = data
            Xs = Xs.to(device)
            Xt = Xt.to(device)
            # embed = embed.to(device)
            with torch.no_grad():
                embed, Xs_feats = arcface(
                    F.interpolate(
                        Xs[:, :, 19:237, 19:237],
                        [112, 112],
                        mode="bilinear",
                        align_corners=True,
                    )
                )
            same_person = same_person.to(device)
            # diff_person = (1 - same_person)

            # train G
            opt_G.zero_grad()
            Y, Xt_attr = G(Xt, embed)

            Di = D(Y)
            L_adv = 0

            for di in Di:
                L_adv += hinge_loss(di[0], True)

            Y_aligned = Y[:, :, 19:237, 19:237]
            ZY, Y_feats = arcface(
                F.interpolate(
                    Y_aligned, [112, 112], mode="bilinear", align_corners=True
                )
            )
            L_id = (1 - torch.cosine_similarity(embed, ZY, dim=1)).mean()

            Y_attr = G.get_attr(Y)
            L_attr = 0
            for i in range(len(Xt_attr)):
                L_attr += torch.mean(
                    torch.pow(Xt_attr[i] - Y_attr[i], 2).reshape(batch_size, -1),
                    dim=1,
                ).mean()
            L_attr /= 2.0

            L_rec = torch.sum(
                0.5
                * torch.mean(torch.pow(Y - Xt, 2).reshape(batch_size, -1), dim=1)
                * same_person
            ) / (same_person.sum() + 1e-6)

            lossG = 1 * L_adv + 10 * L_attr + 5 * L_id + 10 * L_rec
            # lossG = 1*L_adv + 10*L_attr + 5*L_id + 10*L_rec
            with amp.scale_loss(lossG, opt_G) as scaled_loss:
                scaled_loss.backward()

            # lossG.backward()
            opt_G.step()

            # train D
            opt_D.zero_grad()
            # with torch.no_grad():
            #     Y, _ = G(Xt, embed)
            fake_D = D(Y.detach())
            loss_fake = 0
            for di in fake_D:
                loss_fake += hinge_loss(di[0], False)

            true_D = D(Xs)
            loss_true = 0
            for di in true_D:
                loss_true += hinge_loss(di[0], True)
            # true_score2 = D(Xt)[-1][0]

            lossD = 0.5 * (loss_true.mean() + loss_fake.mean())

            with amp.scale_loss(lossD, opt_D) as scaled_loss:
                scaled_loss.backward()
            # lossD.backward()
            opt_D.step()
            batch_time = time.time() - start_time
            if iteration % show_step == 0:
                image = make_image(Xs, Xt, Y)
                vis.image(image[::-1, :, :], opts={"title": "result"}, win="result")
                cv2.imwrite("./gen_images/latest.jpg", image.transpose([1, 2, 0]))
            print(f"epoch: {epoch}    {iteration} / {len(dataloader)}")
            print(
                f"lossD: {lossD.item()}    lossG: {lossG.item()} "
                f"batch_time: {batch_time}s"
            )
            print(
                f"L_adv: {L_adv.item()} L_id: {L_id.item()} "
                f"L_attr: {L_attr.item()} L_rec: {L_rec.item()}"
            )
            if iteration % 1000 == 0:
                torch.save(G.state_dict(), "./saved_models/G_latest.pth")
                torch.save(D.state_dict(), "./saved_models/D_latest.pth")
