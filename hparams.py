import tensorflow as tf


# Default hyperparameters:
hparams = tf.contrib.training.HParams(

  # Audio:
  num_mels=80,
  num_freq=1025,
  mcep_dim=24,
  mcep_alpha=0.41,
  minf0=40,
  maxf0=500,
  sample_rate=16000,
  feature_type='mcc', # mcc or melspc
  frame_length_ms=25, #50,
  frame_shift_ms=10, #12.5,
  preemphasis=0.97,
  min_level_db=-100,
  ref_level_db=20,
  noise_injecting=True,

  # Training:
  use_cuda=True,
  use_local_condition=True,
  batch_size=5,
  sample_size=16000,
  learning_rate=2e-4,
  training_steps=200000,
  checkpoint_interval=5000,

  # Model
  n_stacks=11,
  fft_channels=256,
  quantization_channels=256,
)


def hparams_debug_string():
  values = hparams.values()
  hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
  return 'Hyperparameters:\n' + '\n'.join(hp)
