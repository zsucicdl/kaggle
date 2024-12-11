class DebertaConfig:
    model_name = "debertav3base"
    learning_rate = 1.5e-5
    weight_decay = 0.02
    hidden_dropout_prob = 0.005
    attention_probs_dropout_prob = 0.005
    num_train_epochs = 5
    n_splits = 4
    batch_size = 12
    random_seed = 42
    save_steps = 100
    max_length = 512
