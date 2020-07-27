import os
import streamlit as st
import numpy as np
from reface import env
from reface.data_lib import Dataset
from reface.faceshifter.train_AEI import ModelManager, Trainer


env.device = "cpu"
model_names = [
    "imsz256-60_bs8-2-1_G_L6e-04_full_D_L4e-04_lr6_sc3",
    "imsz64-60_bs64-12_G_L4e-04_l6-x0.5_D_L4e-04_lr6_sc3_all-BN",
    # "imsz64-60_bs64-12_G_L4e-03_l6-x0.5_D_L4e-04_lr6_sc3_all-BN",
    # "imsz64-60_bs32-6_G_L4e-04_D_L4e-04_lr6_sc3_all-BN",
]


@st.cache(allow_output_mutation=True)
def get_datasets():
    return Dataset("train"), Dataset("test")


@st.cache(allow_output_mutation=True)
def get_model_manager(model_dir):
    mmgr = ModelManager(model_dir)
    if mmgr.cfg.INPUT.IMAGE_SIZE >= 256:
        n_rows = 2
        n_cols = 1
    elif mmgr.cfg.INPUT.IMAGE_SIZE >= 128:
        n_rows = 4
        n_cols = 2
    else:
        n_rows = 8
        n_cols = 4

    mmgr.cfg.INPUT.LOADER.NUM_WORKERS = 0
    mmgr.cfg.INPUT.MIN_FACE_SIZE = 160
    mmgr.cfg.INPUT.TRAIN.BATCH_SIZE = n_rows
    mmgr.cfg.INPUT.TRAIN.SAME_PERSON_PAIRS_PER_BATCH = 1
    mmgr.cfg.TEST.VIS_MAX_IMAGES = n_rows

    return mmgr, n_rows, n_cols


@st.cache(allow_output_mutation=True)
def get_trainer(model_manager):
    ds_train, ds_test = get_datasets()
    trainer = Trainer(model_manager, ds_train, ds_test)
    trainer.last_ckpt_id = None
    trainer.generator.eval()
    trainer.discriminator.eval()
    return trainer


model_name = st.sidebar.selectbox("Model", model_names)
mmgr, n_rows, n_cols = get_model_manager(os.path.join(".models", model_name))

ckpt_id = st.sidebar.selectbox("Checkpoint", mmgr.list_checkpoints()[::-1])
which_data = st.sidebar.selectbox("Data", ["train", "test"])

trainer = get_trainer(mmgr)

if trainer.last_ckpt_id != ckpt_id:
    mmgr.load_from_checkpoint(trainer.generator, trainer.discriminator, it=ckpt_id)
    trainer.last_ckpt_id = ckpt_id

columns = []
for _ in range(n_cols):
    img = trainer.get_demo(which_data, n_rows)
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    columns.append(img)

full_img = np.concatenate(columns, axis=1)

st.image(full_img)

st.button("next")
