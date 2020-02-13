# coding = 'utf-8'

import glob
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from abc import ABC, abstractmethod
from apex import amp

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


class ModelTrainer(ABC):
    """
    This is a class that simplifies the boilerplate code for PyTorch model training.
    # TODO Finish the introduction
    """

    def __init__(self):
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.tb_writer = None
        self.local_rank = None
        self.train_batch_size = None
        self.per_gpu_train_batch_size = None
        self.n_gpu = None
        self.max_steps = None
        self.gradient_accumulation_steps = None
        self.model_name_or_path = None
        self.seed = None
        self.eval_output_dir = None
        self.per_gpu_eval_batch_size = None
        self.eval_batch_size = None

    @abstractmethod
    def get_input(self, batch):
        pass

    @abstractmethod
    def get_metric(self, y, y_pred):
        pass

    def set_seed(self, seed):
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)

    def set_model(self, model, **kwargs):
        self.model = model
        return self

    def set_optimizer(self, optimizer, **kwargs):
        self.optimizer = optimizer
        return self

    def set_scheduler(self, scheduler, **kwargs):
        self.scheduler = scheduler
        return self



    def train_and_eval(self, train_dataset, eval_data_set, args, prefix = "", **kwargs ):
        self.local_rank = args.local_rank
        self.train_batch_size = args.train_batch_size
        self.per_gpu_train_batch_size = args.per_gpu_train_batch_size
        self.n_gpu = args.n_gpu
        self.max_steps = args.max_steps
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.model_name_or_path = args.model_name_or_path
        self.eval_output_dir = args.eval_output_dir
        self.per_gpu_eval_batch_size = args.per_gpu_train_batch_size

        if self.local_rank in [-1, 0]:
            self.tb_writer = SummaryWriter()

        self.train_batch_size = self.per_gpu_train_batch_size * max(1, self.n_gpu)
        train_sampler = RandomSampler(train_dataset) if self.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        t_total = self.set_up_training(args, train_dataloader)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.train_batch_size
            * args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
        )
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if os.path.exists(self.model_name_or_path):
            # set global_step to gobal_step of last saved checkpoint from model path
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
        )
        self.set_seed(args.seed)
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                self.model.train()

                inputs = self.get_input(batch)
                outputs = self.model(**inputs)
                loss = outputs[0]  # Here we assume the loss function is written already in the model function

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        logs = {}
                        if (
                                args.local_rank == -1 and args.evaluate_during_training
                        ):  # Only evaluate when single GPU otherwise metrics may not average well
                            results = self.evaluate(eval_data_set, prefix, **kwargs)
                            for key, value in results.items():
                                eval_key = "eval_{}".format(key)
                                logs[eval_key] = value

                        loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                        learning_rate_scalar = self.scheduler.get_lr()[0]
                        logs["learning_rate"] = learning_rate_scalar
                        logs["loss"] = loss_scalar
                        logging_loss = tr_loss

                        for key, value in logs.items():
                            self.tb_writer.add_scalar(key, value, global_step)
                        print(json.dumps({**logs, **{"step": global_step}}))

                    if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            self.model.module if hasattr(self.model, "module") else self.model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(self.scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

                if self.max_steps > 0 and global_step > self.max_steps:
                    epoch_iterator.close()
                    break
            if self.max_steps > 0 and global_step > self.max_steps:
                train_iterator.close()
                break

        if args.local_rank in [-1, 0]:
            self.tb_writer.close()

        return global_step, tr_loss / global_step

    def evaluate(self, eval_dataset, prefix="", *args, **kwargs):
        results = dict()

        if not os.path.exists(self.eval_output_dir) and self.local_rank in [-1, 0]:
            os.makedirs(self.eval_output_dir)

        self.eval_batch_size = self.per_gpu_eval_batch_size * max(1, self.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.eval_batch_size)

        # multi-gpu eval
        if self.n_gpu > 1:
            model = torch.nn.DataParallel(self.model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", selfeval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = self.get_input(batch)
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)

        result = self.get_metric(preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(self.eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        return results

    def set_up_training(self, args, train_dataloader):
        if self.max_steps > 0:
            t_total = args.max_steps
            self.num_train_epochs = self.max_steps // (len(train_dataloader) // self.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.gradient_accumulation_steps * self.num_train_epochs
        if os.path.isfile(os.path.join(self.model_name_or_path, "optimizer.pt")) and os.path.isfile(
                os.path.join(self.model_name_or_path, "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
            self.scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=args.fp16_opt_level)
        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        # Distributed training (should be after apex fp16 initialization)
        if args.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
            )
        return t_total
