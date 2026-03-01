import numpy as np


def base_lengths(train_end_t, dt, future_steps):
    n_train = int(np.round(train_end_t / dt))
    n_total = n_train + future_steps
    return n_train, n_total
