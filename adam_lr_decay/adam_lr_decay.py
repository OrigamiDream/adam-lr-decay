import re
from typing import Union

try:
    import tensorflow as tf
    from tensorflow.keras import optimizers
except ImportError as e:
    raise ImportError('You must install `tensorflow>=2.11` or `tensorflow-cpu>=2.11`: {}'.format(e))

from packaging import version

if version.parse(tf.__version__) < version.parse('2.11.0'):
    raise version.InvalidVersion(
        'The module must be `tensorflow>=2.11`, but current tensorflow version is `{}`'.format(tf.__version__)
    )


class AdamLRDecay(optimizers.Adam):
    def __init__(self,
                 learning_rate: Union[float, tf.Tensor, optimizers.schedules.LearningRateSchedule] = 0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 amsgrad=False,
                 weight_decay=None,
                 clipnorm=None,
                 clipvalue=None,
                 global_clipnorm=None,
                 use_ema=False,
                 ema_momentum=0.99,
                 ema_overwrite_frequency=None,
                 jit_compile=True,
                 name='AdamLRDecay',
                 **kwargs):
        super().__init__(learning_rate=learning_rate,
                         beta_1=beta_1,
                         beta_2=beta_2,
                         epsilon=epsilon,
                         amsgrad=amsgrad,
                         weight_decay=weight_decay,
                         clipnorm=clipnorm,
                         clipvalue=clipvalue,
                         global_clipnorm=global_clipnorm,
                         use_ema=use_ema,
                         ema_momentum=ema_momentum,
                         ema_overwrite_frequency=ema_overwrite_frequency,
                         jit_compile=jit_compile,
                         name=name,
                         **kwargs)

    def apply_layerwise_lr_decay(self, var_dict=None, var_name_dicts=None):
        if hasattr(self, '_built') and self._built:
            raise ValueError(
                '`apply_layerwise_lr_decay()` can only be configured before the optimizer is built.'
            )

        if var_dict:
            self._mapped_layerwise_lr_decay = {
                self._var_key(variable): decay_rate for variable, decay_rate in var_dict.items()
            }
        else:
            self._mapped_layerwise_lr_decay = {}
        self._mapped_layerwise_lr_decay_names = var_name_dicts or {}

    def _use_layerwise_lr_decay(self, variable):
        mapped_layerwise_lr_decay = getattr(
            self, '_mapped_layerwise_lr_decay', {}
        )
        mapped_layerwise_lr_decay_names = getattr(
            self, '_mapped_layerwise_lr_decay_names', {}
        )
        variable_id = self._var_key(variable)
        for mapped_id, rate in mapped_layerwise_lr_decay.items():
            if variable_id == mapped_id:
                return rate
        for name, rate in mapped_layerwise_lr_decay_names.items():
            if re.search(name, variable.name) is not None:
                return rate
        return None

    def _calculate_learning_rate(self, variable):
        lr = tf.cast(self.learning_rate, variable.dtype)
        decay_rate = self._use_layerwise_lr_decay(variable)
        if decay_rate is not None:
            decay_rate = tf.cast(decay_rate, variable.dtype)
            return lr * (1. - decay_rate)
        return lr

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = self._calculate_learning_rate(variable)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(self.beta_2, variable.dtype), local_step)

        var_key = self._var_key(variable)
        m = self._momentums[self._index_dict[var_key]]
        v = self._velocities[self._index_dict[var_key]]

        alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            m.assign_add(-m * (1 - self.beta_1))
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1 - self.beta_1), gradient.indices
                )
            )
            v.assign_add(-v * (1 - self.beta_2))
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - self.beta_2),
                    gradient.indices,
                    )
            )
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))
        else:
            # Dense gradients.
            m.assign_add((gradient - m) * (1 - self.beta_1))
            v.assign_add((tf.square(gradient) - v) * (1 - self.beta_2))
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))

    def _apply_weight_decay(self, variables):
        if self.weight_decay is None:
            return
        for variable in variables:
            if self._use_weight_decay(variable):
                lr = self._calculate_learning_rate(variable)
                wd = tf.cast(self.weight_decay, variable.dtype)
                variable.assign_sub(variable * wd * lr)

    def get_config(self):
        return super().get_config()
