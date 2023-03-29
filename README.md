# Adam Layer-wise LR Decay

In [ELECTRA](https://arxiv.org/abs/2003.10555), 
which had been published by Stanford University and Google Brain, 
they had used Layerwise LR Decay technique for the Adam optimizer to prevent Catastrophic forgetting of Pre-trained model.

This repo contains the implementation of Layer-wise LR Decay for Adam, with new Optimizer API that had been proposed in TensorFlow 2.11.

## Usage
```bash
$ pip install adam-lr-decay
```
```python
from tensorflow.keras import layers, models
from adam_lr_decay import AdamLRDecay

# ... prepare training data

# model definition
model = models.Sequential([
    layers.Dense(3, input_shape=(2,), name='hidden_dense'),
    layers.Dense(1, name='output')
])

# optimizer definition with layerwise lr decay
adam = AdamLRDecay(learning_rate=1e-3)
adam.apply_layerwise_lr_decay(var_name_dicts={
    'hidden_dense': 0.1,
    'output': 0.
})
# this config decays the key layers by the value, 
# which is (lr * (1. - decay_rate))

# compile the model
model.compile(optimizer=adam)

# ... training loop
```

In official [ELECTRA repo](https://github.com/google-research/electra/blob/8a46635f32083ada044d7e9ad09604742600ee7b/model/optimization.py#L181),
they have defined the decay rate in the code. The adapted version is as follows:
```python
import collections
from adam_lr_decay import AdamLRDecay

def _get_layer_lrs(layer_decay, n_layers):
    key_to_depths = collections.OrderedDict({
        '/embeddings/': 0,
        '/embeddings_project/': 0,
        'task_specific/': n_layers + 2,
    })
    for layer in range(n_layers):
        key_to_depths['encoder/layer_' + str(layer) + '/'] = layer + 1
    return {
        key: 1. - (layer_decay ** (n_layers + 2 - depth))
        for key, depth in key_to_depths.items()
    }

# ... ELECTRA model definition

adam = AdamLRDecay(learning_rate=1e-3)
adam.apply_layerwise_lr_decay(var_name_dicts=_get_layer_lrs(0.9, 8))

# ... custom training loop
```

The generated decay rate must be looked like this. `0.0` means there is no decay and `1.0` means it is zero learning rate. (non-trainable)
```json
{
  '/embeddings/': 0.6513215599,
  '/embeddings_project/': 0.6513215599, 
  'task_specific/': 0.0, 
  'encoder/layer_0/': 0.6125795109999999, 
  'encoder/layer_1/': 0.5695327899999999, 
  'encoder/layer_2/': 0.5217030999999999, 
  'encoder/layer_3/': 0.46855899999999995, 
  'encoder/layer_4/': 0.40950999999999993, 
  'encoder/layer_5/': 0.3439, 
  'encoder/layer_6/': 0.2709999999999999, 
  'encoder/layer_7/': 0.18999999999999995
}
```

## Citation
```bibtex
@article{clark2020electra,
  title={Electra: Pre-training text encoders as discriminators rather than generators},
  author={Clark, Kevin and Luong, Minh-Thang and Le, Quoc V and Manning, Christopher D},
  journal={arXiv preprint arXiv:2003.10555},
  year={2020}
}
```
