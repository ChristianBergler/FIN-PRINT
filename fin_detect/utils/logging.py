"""
Module: logging.py
Authors: Christian Bergler, Alexander Gebhard, Hendrik Schroeter
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 25.01.2022
"""

import os
import queue
import torch
import logging
import logging.handlers

from typing import Union


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

"""
Logger in order to track training, validation, and testing of the network
"""
class Logger(metaclass=Singleton):
    def __init__(self, name, debug=False, log_dir=None, do_log_name=False):
        level = logging.DEBUG if debug else logging.INFO
        fmt = "%(asctime)s"
        if do_log_name:
            fmt += "|%(name)s"
        fmt += "|%(levelname).1s|%(message)s"
        formatter = logging.Formatter(fmt=fmt, datefmt="%H:%M:%S")

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        handlers = [sh]

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            fs = logging.FileHandler(os.path.join(log_dir, name + ".log"))
            fs.setFormatter(formatter)
            handlers.append(fs)

        self._queue = queue.Queue(1000)
        self._handler = logging.handlers.QueueHandler(self._queue)
        self._listener = logging.handlers.QueueListener(self._queue, *handlers)

        self._logger = logging.getLogger(name)
        self._logger.propagate = False

        self._logger.setLevel(level)
        self._logger.addHandler(self._handler)
        self._listener.start()

    def close(self):
        self._listener.stop()
        self._handler.close()

    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def epoch(
        self,
        phase,
        epoch,
        num_epochs,
        loss,
        accuracy=None,
        f1_score=None,
        precision=None,
        recall=None,
        lr=None,
        epoch_time=None,
    ):
        s = "{}|{:03d}/{:d}|loss:{:0.3f}".format(
            phase.upper().rjust(5, " "), epoch, num_epochs, loss
        )
        if accuracy is not None:
            s += "|acc:{:0.3f}".format(accuracy)
        if f1_score is not None:
            s += "|f1:{:0.3f}".format(f1_score)
        if precision is not None:
            s += "|pr:{:0.3f}".format(precision)
        if recall is not None:
            s += "|re:{:0.3f}".format(recall)
        if lr is not None:
            s += "|lr:{:0.2e}".format(lr)
        if epoch_time is not None:
            s += "|t:{:0.1f}".format(epoch_time)

        self._logger.info(s)

    """
    Writes scalar summary per partition including total_loss (mse loss distance, x, y, w, h + bce loss class, 
    object confidence, no object confidence), x mse loss, y mse loss, width mse loss, height mse loss,
    bce loss obj/no-obj loss, bce loss class confidence, class accuracy, recall 50, recall 75, precision, object
    confidence score, no object confidence score, grid size
    """
    def write_scalar_summaries_logs(
            self,
            summary_writer,
            loss: float,
            metrics: Union[list, dict] = [],
            lr: float = None,
            epoch_time: float = None,
            data_loading_time: float = None,
            batch=None,
            epoch=None,
            phase="train",
    ):
        with torch.no_grad():
            log_str = phase
            if epoch is not None:
                log_str += "|epoch:{}".format(epoch)
            if batch is not None:
                log_str += "|batch:{}".format(batch)
            if loss is not None:
                summary_writer.add_scalar(phase + "/epoch_loss", loss, epoch)
                log_str += "|loss:{:0.3f}".format(loss)
            for tag, value in metrics:
                if batch is not None:
                    summary_writer.add_scalar(phase + "/metric_" + str(tag), value, batch)
                else:
                    summary_writer.add_scalar(phase + "/metric_" + str(tag), value, epoch)
                log_str += "|m_{}:{:0.3f}".format(tag, value)
            if lr is not None:
                summary_writer.add_scalar("lr", lr, batch)
                log_str += "|lr:{:0.2e}".format(lr)
            if epoch_time is not None:
                summary_writer.add_scalar(phase + "/time", epoch_time, epoch)
                log_str += "|t:{:0.1f}".format(epoch_time)
            if data_loading_time is not None:
                summary_writer.add_scalar(
                    phase + "/data_loading_time", data_loading_time, batch
                )
            self.info(log_str)
            summary_writer.flush()
