import json
import torch
from os import path
from sacred import Experiment
import yaml
from sacred.observers import FileStorageObserver
from sacred.utils import recursive_update

from data_utils import load_data
from mlutils.pt.training import Trainer, extend_config_reference
from experiments.utils import EvaluationCallback
from mlutils.exp import yaml_load, yaml_dump

exp = Experiment('exp')

exp.add_config({
    'config_file': 'data/config/ntm.yaml',
    'update': {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
})


@exp.main
def main(config_file, update, _run, _log, _config):
    working_dir = _run.observers[0].dir
    config = yaml_load(config_file)
    recursive_update(config, update)
    yaml_dump(config, path.join(working_dir, 'config.yaml'))
    _config = config
    print(_config)
    print(working_dir)
    dataset = _config['dataset']
    dataset['device'] = update['device']
    train_loader, dev_loader, test_loader, vocab = load_data(**dataset)
    vocab_size = len(vocab.itos)

    # model
    _config['hidden']['features'][0] = vocab_size

    # trainer batch
    test_sample = _config['trainer_batch']['test_sample']
    _config['trainer_batch']['test_sample'] = 1

    config = extend_config_reference(_config)
    trainer = config['trainer']
    trainer['evaluate_interval'] = len(train_loader) * trainer['evaluate_interval']
    trainer['save_checkpoint_interval'] = trainer['evaluate_interval']
    trainer['base_dir'] = working_dir
    yaml_dump(trainer, path.join(working_dir, 'trainer.yaml'))
    trainer['train_iterator'] = train_loader
    trainer['dev_iterator'] = dev_loader
    trainer['test_iterator'] = None
    callback = EvaluationCallback(working_dir, vocab,
                                  corpus_dir=path.join(dataset['data_dir'], 'corpus'),
                                  **config['callback'])
    trainer['callbacks'] = callback
    trainer['logger'] = _log

    print(config)
    trainer = Trainer.from_config(trainer)
    _log.info("model architecture")
    print(trainer.trainer_batch.model)

    trainer.train()

    # testing and save results
    trainer.dev_iterator = test_loader
    trainer.trainer_batch.test_sample = test_sample  # test using many samples, but not in development dataset
    trainer.restore_from_basedir(best=True)
    stat = trainer._evaluate_epoch().get_dict()
    callback.evaluate_topic_coherence()  # topic coherence of best checkpoint
    stat.update(callback.get_dict())
    yaml_dump(stat, path.join(working_dir, 'result.yaml'))
    _log.info('test result of best evaluation {}'.format(stat))


if __name__ == "__main__":
    exp.run_commandline()


