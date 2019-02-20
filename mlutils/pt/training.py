import torch
from torch.nn.utils import clip_grad_value_
import os
from os import path
import math
import sys
import re
import numpy as np
import json
from importlib import import_module
import six
import logging
import copy
from tensorboardX import SummaryWriter
from sklearn.metrics import classification_report

from ..callbacks import CallbackList


def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


class RunningStatistics:

    def __init__(self):
        self.statistics = None

    def add(self, stat):
        """将每个batch获得的统计数据进行累积
        stat: 字典，key为对应的统计量 value 是一个scalar 或者 字典包含value和weight
        """
        if self.statistics is None:
            self.statistics = {}
            for k, v in stat.items():
                if isinstance(v, dict):
                    assert 'value' in v and 'weight' in v, 'value weight should be in {}'.format(str(v))
                    self.statistics[k] = v
                else:
                    self.statistics[k] = {'value': v, 'weight': 1}
        else:
            for k, v in stat.items():
                if not isinstance(v, dict):
                    v = {'value': v, 'weight': 1}
                last_v = self.statistics[k]['value']
                last_w = self.statistics[k]['weight']
                self.statistics[k]['value'] = (v['value'] * v['weight'] + last_v * last_w) / (last_w + v['weight'])
                self.statistics[k]['weight'] += v['weight']
        return self

    def description(self, prefix=''):
        """转化成字符串供输出"""
        if self.statistics is None:
            return 'None'
        return ' | '.join(['{} v{:.6f} w{:.6f}'.format(prefix + k, v['value'], v['weight'])
                           for k, v in self.statistics.items()])

    def get_value(self, k):
        """得到对应k的累值"""
        return self.statistics[k]['value']

    def get_dict(self):
        if self.statistics is None:
            return {}
        return {k: v['value'] for k, v in self.statistics.items()}


def eval_metric_cmp_key(key='loss', cmp=np.less):
    """Return true if new is better than old.
    key: loss, cmp: np.less
    key: accuracy cmp: np.greater
    """
    return lambda new, old: cmp(new.get_value(key), old.get_value(key))


###################################################
# checkpoints
###################################################


def save_checkpoint(filename, model, statistics, logger=None):
    if logger:
        logger.info('save checkpoint to ' + filename)
    if filename and model:
        statistics['state_dict'] = model.state_dict()
        torch.save(statistics, filename)


def load_checkpoint(filename, model=None, logger=None):
    if logger:
        logger.info('load checkpoint from ' + filename)
    statistics = torch.load(filename)
    if model:
        model.load_state_dict(statistics['state_dict'])
    return statistics


###################################################
# metrics
###################################################


def classification_accuracy(pred, gold_true):
    """pred, gold_ture: torch tensors or numpy tensors."""
    if isinstance(pred, torch.Tensor):
        arg_max = torch.argmax(pred, dim=-1, keepdim=False)
        return (arg_max == gold_true).float().mean().item()
    return (np.argmax(pred, axis=-1) == gold_true).mean()


###################################################
# config ralated
###################################################


def parse_class(dotted_path: str):
    """
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.
    """
    try:
        module_path, class_name = dotted_path.rsplit('.', 1)
    except ValueError:
        msg = "%s doesn't look like a module path" % dotted_path
        six.reraise(ImportError, ImportError(msg), sys.exc_info()[2])

    module = import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        msg = 'Module "%s" does not define a "%s" attribute/class' % (
            module_path, class_name)
        six.reraise(ImportError, ImportError(msg), sys.exc_info()[2])


def parse_config_type(config):
    """返回config指定的类或函数对象"""
    if 'type' in config:
        return parse_class(config['type'])
    return eval(config['eval'])


def build_from_config(config):
    """build classes or functions specified in config.
    config file format:
    1. callable
    {
    type: classTypeStr(or function)
    args: ...
    }
    2. just eval
    {
    eval: classTypeStr
    }  -> eval(classTypeStr): useful when class name or function name is needed
    """
    if 'type' not in config:
        assert 'eval' in config, 'eval not in config {}' .format(str(config))
        return eval(config['eval'])
    for k in config.keys():
        if isinstance(config[k], dict):
            config[k] = build_from_config(config[k])
    return parse_class(config.pop('type'))(**config)


def extend_config_reference(config):
    """Extend the reference in config. Make sure that no circular reference in config.
    config:
    {'a': 'b',
    'b': {}} ->
    {'a': {}(which is denoted by 'b'),
    'b': {}}
    """
    def _parse_reference(keys, r):
        if hasattr(r, '__getitem__'):
            try:
                v = r.__getitem__(keys)
                return v
            except (KeyError, TypeError, IndexError):
                pass
        if isinstance(keys, tuple):
            v = _parse_reference(keys[0], r)
            if v is not None:
                if len(keys) == 1:
                    return v
                return _parse_reference(keys[1:], v)
        return None

    def _sub_reference(cf, ori):
        it = cf.keys() if isinstance(cf, dict) else range(len(cf))
        for k in it:
            v = cf[k]
            if isinstance(v, (dict, list)):
                v = _sub_reference(v, ori)
            else:
                r = _parse_reference(v, ori)
                if r is not None:
                    v = r
            cf[k] = v
        return cf

    replace = copy.deepcopy(config)
    return _sub_reference(replace, replace)



class TrainerBatch:
    """每个batch数据和模型的操作"""
    def __init__(self,
                 model,
                 optimizer,
                 criterion,
                 data_attrs,  # List of attributions of a batch of data. The order is important.
                 label_attr,
                 device,
                 grad_clip_value=0,
                 ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.data_attrs = data_attrs if isinstance(data_attrs, list) else [data_attrs]
        self.label_attr = label_attr
        self.dst_device = device
        self.grad_clip_value = grad_clip_value

    def place_data_to_device(self, x):
        if x.device != self.dst_device:
            return x.to(self.dst_device)
        return x

    def attrs_from_batch(self, batch, attr):
        single = False
        if not isinstance(attr, list):
            single = True
            attr = [attr]
        if isinstance(batch, list):
            batch = [self.place_data_to_device(b) for b in batch]
        elif isinstance(batch, dict):
            batch = [self.place_data_to_device(batch[d]) for d in attr]
        else:
            batch = [self.place_data_to_device(getattr(batch, d)) for d in attr]
        return batch if not single else batch[0]

    def train_batch(self, data, train=True):
        self.model.train(train)

        x = self.attrs_from_batch(data, self.data_attrs)
        out = self.model(*x)
        y_true = self.attrs_from_batch(data, self.label_attr)
        loss = self.criterion(out, y_true)
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip_value > 0:
                clip_grad_value_(self.model.parameters(), self.grad_clip_value)
            self.optimizer.step()
        batch_size = len(y_true)
        return {'loss':
                    {'value': loss.item(),
                     'weight': batch_size
                     },
                'accuracy':
                    {'value': classification_accuracy(out, y_true),
                     'weight': batch_size
                     }
                }

    def eval_batch(self, data):
        return self.train_batch(data, train=False)

    def predict_batch(self, data, prob):
        """if not prob, return the argmax with dim = -1"""
        self.model.eval()
        with torch.no_grad():
            x = self.attrs_from_batch(data, self.data_attrs)
            out = self.model(*x)
        if not prob:
            return torch.argmax(out, dim=-1)
        return torch.softmax(out, dim=-1)

    @staticmethod
    def from_config(config):
        config['model'] = build_from_config(config['model']).to(config['device'])
        if 'type' in config['optimizer']:
            optimizer = parse_class(config['optimizer'].pop('type'))
            optimizer = optimizer(config['model'].parameters(), **config['optimizer'])
        else:
            optimizer = eval(config['optimizer']['eval'])
        config['optimizer'] = optimizer
        return build_from_config(config)


class Trainer:
    """使用logging和tensorboardX SummaryWriter.
    基础功能：checkpoint loading ans saving, summaries during training, data iteration, logging, write training results.
    """
    def __init__(self,
                 base_dir,
                 train_iterator,
                 num_epochs,
                 trainer_batch,
                 dev_iterator=None,
                 test_iterator=None,
                 test_report=classification_report,  # report about (y_true, y_pred)
                 eval_metric=eval_metric_cmp_key(),  # True if (New, Old) -> True
                 statistics=RunningStatistics,
                 early_stop=None,
                 evaluate_interval=None,
                 save_checkpoint_interval=None,
                 num_checkpoints_keep=10,
                 print_statistics_interval=None,
                 callbacks=None,
                 logger=None):
        os.makedirs(base_dir, exist_ok=True)
        self.base_dir = base_dir
        if logger is None:
            logger = get_logger(path.join(base_dir, 'log'))
        self.logger = logger
        self.train_iterator = train_iterator
        self.dev_iterator = dev_iterator
        self.test_iterator = test_iterator
        self.test_report = test_report
        self.num_epochs = num_epochs
        self.trainer_batch = trainer_batch
        self.save_checkpoint_interval = math.inf if save_checkpoint_interval is None else save_checkpoint_interval
        self.print_statistics_interval = len(train_iterator) if print_statistics_interval is None \
            else print_statistics_interval
        self.evaluate_interval = len(train_iterator) if evaluate_interval is None else evaluate_interval
        self.eval_metric = eval_metric
        self.best_eval = None
        self.early_stop = early_stop
        self._early_stop_counter = 0
        self.statistics = statistics
        self.num_checkpoints_keep = num_checkpoints_keep
        self.checkpoints = []
        self.summary_writer = SummaryWriter(path.join(base_dir, 'summary'))

        callbacks = callbacks or []
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        self.callbacks = CallbackList(callbacks)
        self.callbacks.set_trainer(self)

    def _write_summary(self, stat, step, prefix='train'):
        stat = stat.get_dict()
        for k, v in stat.items():
            self.summary_writer.add_scalars(k, {prefix: v}, global_step=step)

    def load_checkpoint(self, model_path):
        return load_checkpoint(model_path, self.trainer_batch.model, self.logger)

    def restore_from_basedir(self, latest=False, best=False):
        """Restore from latest or best model."""
        if latest:
            ps = []
            for p in os.listdir(self.base_dir):
                r = re.match(r'model_([\d]+).pt', p)
                if r:
                    ps.append((int(r.group(1)), p))
            if len(ps) == 0:
                self.logger.warning('use latest model to restore, but not found any. Try to use best evaluation modle')
            else:
                ps = sorted(ps, key=lambda x: x[0])
                return self.load_checkpoint(path.join(self.base_dir, ps[-1][1]))

        if best:
            p = path.join(self.base_dir, 'model_best.pt')
            if not path.exists(p):
                self.logger.warning('best model {} does not exist'.format(p))
            else:
                return self.load_checkpoint(p)

        return None

    def _evaluate_epoch(self):
        stat = self.statistics()
        for batch in self.dev_iterator:
            stat.add(self.trainer_batch.eval_batch(batch))
        return stat

    def _save_checkpoint(self, statistics, best):
        if best:
            filename = path.join(self.base_dir, 'model_best.pt')
        else:
            filename = path.join(self.base_dir, 'model_{}.pt'.format(statistics['batch_no']))
            self.checkpoints.append(filename)

        save_checkpoint(filename, self.trainer_batch.model, statistics, self.logger)
        if len(self.checkpoints) > self.num_checkpoints_keep:
            to_delete = self.checkpoints[0]
            if path.exists(to_delete):
                os.remove(to_delete)
            self.checkpoints = self.checkpoints[1:]

    def _save_best_statistics(self):
        save_path = path.join(self.base_dir, 'best_eval.json')
        stat = self.statistics() if self.best_eval is None else self.best_eval
        self.logger.info('save best evaluation statistics to {}'.format(save_path))
        with open(save_path, 'w') as f:
            json.dump(stat.get_dict(), f)

    def train(self, restore=True):
        # restoration
        batch_no = 0
        if restore:
            stat = self.restore_from_basedir(latest=True, best=True)
            if not stat:
                self.logger.warning('No checkpoint found and start training from fresh model')
            else:
                batch_no = stat['batch_no']
                self.logger.info('found checkpoint at batch_no {}'.format(batch_no))
                if self.dev_iterator:
                    self.logger.info('try to evaluate checkpoint')
                    self.best_eval = self._evaluate_epoch()
                    self.logger.info('checkpoint performance: {}'.format(self.best_eval.description('eval_')))

        self.callbacks.on_train_begin()
        total_batch_no = len(self.train_iterator) * self.num_epochs + batch_no
        statistics = self.statistics()
        for e in range(1, 1 + self.num_epochs):
            self.callbacks.on_epoch_begin(e)
            for b, batch in enumerate(self.train_iterator):
                # train batch
                self.callbacks.on_batch_begin(b)
                res = self.trainer_batch.train_batch(batch)
                statistics.add(res)
                batch_no += 1

                # save checkpoint
                if batch_no % self.save_checkpoint_interval == 0 or batch_no == total_batch_no:
                    self.logger.info('saving checkpoint')
                    # save checkpoint
                    self._save_checkpoint({'batch_no': batch_no}, False)

                # print statistics
                if batch_no % self.print_statistics_interval == 0 or batch_no == total_batch_no:
                    self.logger.info('train {}/{}(epoch {}/{}, {:.2f}), {}'.format(batch_no, total_batch_no, e,
                                                                                   self.num_epochs,
                                                                                   batch_no / total_batch_no,
                                                                                   statistics.description('train_')))
                    self._write_summary(statistics, batch_no, 'train')
                    if batch_no == total_batch_no:
                        self.logger.info('finished training')
                        self._save_best_statistics()
                        self.test_performance()

                    statistics = self.statistics()

                # evaluating
                if self.dev_iterator and batch_no % self.evaluate_interval == 0:
                    self.logger.info('evaluating model performance')
                    res = self._evaluate_epoch()
                    self.logger.info('evaluate {}/{}(epoch {}/{}, {:.2f}), {}'.format(batch_no, total_batch_no, e,
                                                                                      self.num_epochs,
                                                                                      batch_no / total_batch_no,
                                                                                      res.description('eval_')))
                    self._write_summary(res, batch_no, 'eval')
                    # save best
                    if self.best_eval is None or self.eval_metric(new=res, old=self.best_eval):
                        self.logger.info('new best evaluation {}'.format(res.description('eval_')))
                        self.best_eval = res
                        stat = res.get_dict()
                        stat['batch_no'] = batch_no
                        self._save_checkpoint(stat, True)
                        self._early_stop_counter = 0
                    else:
                        # early stop
                        if self.early_stop:
                            self._early_stop_counter += 1
                            if self._early_stop_counter >= self.early_stop:
                                self.logger.info('early stop toggled and finished training')
                                self._save_best_statistics()
                                self.test_performance()
                                self.callbacks.on_batch_end(b)
                                self.callbacks.on_epoch_end(e)
                                self.callbacks.on_train_end()
                                return
                self.callbacks.on_batch_end(b)
            self.callbacks.on_epoch_end(e)
        self.callbacks.on_train_end()

    def predict(self, test_iterator, prob):
        out = []
        for data in test_iterator:
            out.append(self.trainer_batch.predict_batch(data, prob))
        return torch.cat(out, dim=0)

    def test_performance(self, test_iterator=None):
        test_iterator = test_iterator if test_iterator is not None else self.test_iterator
        if test_iterator is not None:
            self.logger.info('begin to test model performance')
            y_true = []
            y_pred = []
            for data in test_iterator:
                t = self.trainer_batch.attrs_from_batch(data, self.trainer_batch.label_attr)
                if isinstance(t, torch.Tensor):
                    t = t.detach().cpu().numpy()
                y_true.append(t)
                t = self.trainer_batch.predict_batch(data, prob=False)
                if isinstance(t, torch.Tensor):
                    t = t.detach().cpu().numpy()
                y_pred.append(t)
            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)
            report = self.test_report(y_true=y_true, y_pred=y_pred)
            self.logger.info('test performance:\n{}'.format(report))
            return report
        return 'No report'

    @staticmethod
    def from_config(config):
        config['trainer_batch'] = parse_class(config['trainer_batch']['type']).from_config(config['trainer_batch'])
        return build_from_config(config)
