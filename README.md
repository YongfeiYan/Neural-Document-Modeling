# Neural-Document-Modeling

Simple PyTorch implementation of models
- NVDM in [Neural Variational Inference for Text Processing][1]. Yishu Miao, Lei Yu, Phil Blunsom. ICML 2016.
- GSM in [Discovering Discrete Latent Topics with Neural Variational Inference][2]. Yishu Miao, Edward Grefenstette, Phil Blunsom
- NTM and NTMR in [Coherence-Aware Neural Topic Modeling][3] Ran Ding, Ramesh Nallapati, Bing Xiang. EMNLP 2018



References to other related implementations:

- [Pytorch AVITM](https://github.com/hyqneuron/pytorch-avitm)
- [autoencoding_vi_for_topic_models](https://github.com/akashgit/autoencoding_vi_for_topic_models)  data/20news-clean was obtained there.
- [nvdm](https://github.com/ysmiao/nvdm)



## Dependencies

1. To evaluate topic coherence, the [topic interpretationbility](https://github.com/jhlau/topic_interpretability) toolkit should be used which was already downloaded into ./scripts directory. To evaluate topic coherence, follow the steps below.

   ```bash
   # Setup a Python 2 environment named py2 using conda package manager.
   conda create -n py2 --file data/py2.env
   # Run customized script at scripts/topic_coherence.sh
   # bash scripts/topic_coherence.sh topic_file corpus_dir save_prefix
   bash scripts/topic_coherence.sh data/topic-1.topics data/20news-clean/corpus/ data/topic-1.res
   ```

2. The codes dependent on 

   ```bash
   python 3.6
   pytorch 1.0.0
   sacred 0.7.0
   scikit-learn 0.19.1
   ```

    

## Models

Run the model with different config files

```bash
export PYTHONPATH=`pwd`:
# any config of: nvdm.yaml, gsm.yaml, ntm.yaml, ntmr.yaml
python experiments/ntm.py -F data/exp/ntm with config_file=data/config/nvdm.yaml 
# Use WETC topic coherence measure as in Coherence-Aware Neural Topic Modeling
python experiments/ntm.py -F data/exp/ntm with config_file=data/config/nvdm.yaml update.callback=callback_wetc
```



## Results

The test perplexity and topic coherence on best evaluation checkpoint when topic number is 50.

| Model | PPL        | Topic Coherence |
| ----- | ---------- | --------------- |
| GSM   | 789.27     | 0.212           |
| NTMR  | 818.30     | __0.347__       |
| NTM   | 883.71     | 0.281           |
| NVDM  | __769.50__ | 0.158           |

Please let me known if better results are obtained or any advice on model and implementation details.



## Notes

- The performance of NTMR is dependent on optimizer type.





[1]: https://arxiv.org/abs/1511.06038
[2]:  https://arxiv.org/abs/1706.00359
[3]: https://arxiv.org/abs/1809.02687