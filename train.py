#! /usr/bin/env python
import argparse
import logging
from face_swap.data_lib import Dataset
from face_swap.faceshifter.train_AEI import ModelManager, Trainer


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("model_dir")
    opts = cli.parse_args()
    logging.basicConfig(level=logging.INFO)
    mmgr = ModelManager(opts.model_dir)
    ds_train = Dataset("train")
    ds_test = Dataset("test")
    trainer = Trainer(mmgr, ds_train, ds_test, visdom_addr="http://172.31.45.229")

    trainer.train()


if __name__ == '__main__':
    main()
