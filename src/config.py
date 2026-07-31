
GPT2_SMALL = {
    'emb_dim': 768,
    'vocab_size': 50527,
    'context_length': 1024,
    'num_heads': 12,
    'num_layers': 12,
    'drop_rate': 0.1,
    'qkv_bias': True,
}

TRAIN_CONFIG_BKP = {
    'max_tokens': 100_000_000,
    'context_length': 256,
    'learning_rate': 3e-4,
    'min_learning_rate': 3e-5,
    'warmup_steps': 1000,
    'gradient_accumalation_steps': 16,
    'weight_decay': 0.1,
    'max_gradient_norm': 1.0,
    'validation_batches': 20,
    'evaluation_interval': 500,
    'checkpoint_interval': 500,
    'log_interval': 10,
    'early_stopping_patience': 5,
    'early_stopping_min_delta': 0.001,
    'checkpoint_directory': 'checkpoints',
    'batch_size': 1,
}


TRAIN_CONFIG = {
    'max_tokens': 100_000,
    'context_length': 256,
    'learning_rate': 3e-4,
    'min_learning_rate': 3e-5,
    'warmup_steps': 5,
    'gradient_accumalation_steps': 16,
    'weight_decay': 0.1,
    'max_gradient_norm': 1.0,
    'validation_batches': 5,
    'evaluation_interval': 10,
    'checkpoint_interval': 10,
    'log_interval': 1,
    'early_stopping_patience': 5,
    'early_stopping_min_delta': 0.001,
    'checkpoint_directory': 'checkpoints',
    'batch_size': 1,
}