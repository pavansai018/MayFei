
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
    'context_length': 1024,
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
    'enable_last_checkpoint_callback': True,
    'enable_learning_rate_callback': True,
    'enable_best_checkpoint_callback': True,
    'enable_early_stopping_callback': True,
    'phase_start_checkpoint': 'checkpoints/mayfei_120m.ckpt',
}


TRAIN_CONFIG = {
    'max_tokens': 380_000_000,
    'context_length': 1024,
    'learning_rate': 1e-4,
    'min_learning_rate': 1e-5,
    'warmup_steps': 200,
    'gradient_accumalation_steps': 16,
    'weight_decay': 0.1,
    'max_gradient_norm': 1.0,
    'validation_batches': 50,
    'evaluation_interval': 250,
    'checkpoint_interval': 1000,
    'log_interval': 10,
    'early_stopping_patience': 5,
    'early_stopping_min_delta': 0.001,
    'checkpoint_directory': 'mayfei_checkpoints',
    'batch_size': 1,
    'enable_last_checkpoint_callback': True,
    'enable_learning_rate_callback': True,
    'enable_best_checkpoint_callback': True,
    'enable_early_stopping_callback': False,
    'phase_start_checkpoint': 'mayfei_checkpoints/mayfei_120m.ckpt',
}


INFERENCE_CONFIG = {
    'context_length': 1024,
    'eos_token_id': 50256,
    'default_checkpoint': 'checkpoints/best.ckpt',
}



MAYFEI_SMALL = {
    'emb_dim': 768,
    'vocab_size': 65519,
    'context_length': 1024,
    'num_heads': 12,
    'num_layers': 12,
    'drop_rate': 0.1,
    'qkv_bias': True,
    'tokenizer_name': 'Skywork/Skywork-13B-base',
}

MAYFEI_INFERENCE_CONFIG = {
    'context_length': 1024,
    'eos_token_id': 2,
    'default_checkpoint': 'mayfei_checkpoints/best.ckpt',
}
