# !/user/bin/python
# -*- coding:utf-8 -*-
"""
date：          2019-07-23
Description :
auther : wcy
"""
# import modules
import os
import sys
from torch import optim
import optim_util as ex_optim

curr_path = os.getcwd()
sys.path.append(curr_path)

__all__ = []

# define class


# define function
def get_optim(params, args):
    """

    :param args:
    :return:
    """
    if args.optimizer == 'adabound':
        return ex_optim.AdaBound(
            params,
            lr=args.lr,
            betas=args.betas,
            final_lr=args.final_lr,
            gamma=args.gamma,
            eps=args.eps,
            weight_decay=args.weight_decay,
            amsbound=args.amsbound)
    if args.optimizer == "adadelta":
        return optim.Adadelta(
            params,
            lr=args.lr,
            rho=args.rho,
            eps=args.eps,
            weight_decay=args.weight_decay)
    elif args.optimizer == "adagrad":
        return optim.Adagrad(
            params,
            lr=args.lr,
            lr_decay=args.lr_decay,
            weight_decay=args.weight_decay,
            initial_accumulator_value=args.initial_accumulator_value)
    elif args.optimizer == "adam":
        return optim.Adam(
            params,
            lr=args.lr,
            betas=args.betas,
            eps=args.eps,
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad)
    elif args.optimizer == "adamax":
        return optim.Adamax(
            params,
            lr=args.lr,
            betas=args.betas,
            eps=args.eps,
            weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        return ex_optim.AdamW(
            params,
            lr=args.lr,
            betas=args.betas,
            eps=args.eps,
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad)
    elif args.optimizer == 'asgd':
        return optim.ASGD(
            params,
            lr=args.lr,
            lambd=args.lambd,
            alpha=args.alpha,
            t0=args.t0,
            weight_decay=args.weight_decay)
    # elif args.optimizer == 'lbfgs':
    # elif args.optimizer == 'lr_scheduler':
    # elif args.optimizer == 'optimizer':
    elif args.optimizer in ['lookahead', 'lookaheadadam']:
        return ex_optim.LookaheadAdam(
            params,
            lr=args.lr,
            betas=args.betas,
            eps=args.eps,
            weight_decay=args.weight_decay,
            alpha=args.alpha,
            k_steps=args.k_steps)
    elif args.optimizer == 'lookaheadsgd':
        return ex_optim.LookaheadSGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            dampening=args.dampening,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
            alpha=args.alpha,
            k_steps=args.ksteps)
    elif args.optimizer == "rmsprop":
        return optim.RMSprop(
            params,
            lr=args.lr,
            alpha=args.alpha,
            eps=args.eps,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            centered=args.centered)
    elif args.optimizer == "rprop":
        return optim.Rprop(
            params,
            lr=args.lr,
            etas=args.etas,
            step_sizes=args.step_sizes)
    elif args.optimizer == "sgd":
        return optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            dampening=args.dampening,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov)
    # elif args.optimizer == "sparse_adam":
    else:
        raise AttributeError(
            "This code is for pytorch version 1.1.0. "
            "You should choose an existing pytorch optimizer. "
            "If you need lbfgs or sparse_adam, please modify this code.")


# main
if __name__ == '__main__':
    class Args:
        def __init__(self):
            self.optimizer = 'lbfgs'
    args = Args()
    get_optim(None, args)