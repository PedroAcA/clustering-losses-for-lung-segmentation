#!/usr/bin/env python

import argparse
from torch import optim as optim
from models import ConvNet
from IO import get_loader, Checkpoint
import torch
from training_and_testing import train, test
from config import cfg as conf


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='Would you like to train, test or train and test the model?',
                        choices=["train", "test", "train_test"], type=str, required=True)
    parser.add_argument('--config_file', help='path of config file', default=None, type=str)
    parser.add_argument('--test_type', help='Would you like to do a qualitative or quantitative test?',
                        choices=["qualitative", "quantitative", "all"],
                        default="quantitative", type=str, required=False)
    parser.add_argument('--test_batch_size', help='Optional. Batch size used for tests', default=1,
                        type=int, required=False)
    parser.add_argument('--chkpt_file', help='Optional. Checkpoint file to load when test mode is enabled', default='',
                        type=str, required=False)
    parser.add_argument('opts', help='modify arguments', default=None, nargs=argparse.REMAINDER)
    return parser


def custom_lr(maxiter, power):
    fnc = lambda epoch: (1 - epoch / maxiter) ** power
    return fnc


def create_optimizers_and_schedulers(cfg, prediction_model):
    lr = cfg.SOLVER.LR
    momentum = cfg.SOLVER.MOMENTUM
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    betas = cfg.SOLVER.BETAS
    available_optims = ['Adam', 'SGD']
    # Optimizer
    if cfg.SOLVER.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(prediction_model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    elif cfg.SOLVER.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(prediction_model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError("Unexpected cfg.SOLVER.OPTIMIZER. Expected {}, but got {} instead.".format(available_optims, cfg.SOLVER.OPTIMIZER))

    if cfg.SOLVER.USE_POLY_DECAY:
        # Scheduler
        poly_decay = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=custom_lr(maxiter=cfg.SOLVER.EPOCHS, power=10))
    else:
        poly_decay = None

    return optimizer, poly_decay


def main():
    # argparse
    parser = create_parser()
    args = parser.parse_args()
    if args.mode!= 'train' and args.test_type=='qualitative':
        assert args.test_batch_size==1, "In order to save images, test_batch_size should be 1, but received {} instead".format(args.test_batch_size)

    # config setup
    if args.config_file is not None:
        conf.merge_from_file(args.config_file)
    if args.opts is not None: conf.merge_from_list(args.opts)
    conf.SYSTEM.DEVICE = conf.SYSTEM.DEVICE if torch.cuda.is_available() else 'cpu'
    conf.freeze()
    # Model
    prediction_model = ConvNet(conf)
    prediction_model = prediction_model.to(conf.SYSTEM.DEVICE)
    optimizer, poly_decay = create_optimizers_and_schedulers(conf, prediction_model)
    # checkpointer
    chkpt = Checkpoint(prediction_model, conf)
    if args.chkpt_file:
        chkpt.load(filepath=args.chkpt_file)

    print("Config {}".format(conf))
    chkpt.log_config()
    if args.mode != "test":
        # load the data
        train_loader = get_loader(conf, 'train')
        train(conf, prediction_model, optimizer, train_loader, chkpt, poly_decay=poly_decay)
        chkpt.save()
    if args.mode != "train":
        print("Test type {}".format(args.test_type))
        test_loader = get_loader(conf, 'test', test_batch_size=args.test_batch_size)
        samples = ['JPCNN075', 'JPCNN037', 'JPCLN027', 'JPCLN038']
        test(conf, prediction_model, test_loader, args.test_type, chkpt, samples)


if __name__ == "__main__":
    main()
