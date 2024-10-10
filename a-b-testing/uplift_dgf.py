from itertools import product

import numpy as np


def get_daily_visitors(days, lam, lam_noise, rng):
    noise = rng.poisson(lam_noise, days) * rng.choice([1, -1], days)
    daily_visitors = rng.poisson(lam, days) + noise
    return daily_visitors


def get_daily_conversions(daily_visitors, conversion_rate, noise_alpha, noise_beta, rng):
    n_days = len(daily_visitors)
    noise = rng.choice([1, -1], n_days) * rng.beta(noise_alpha, noise_beta, n_days)
    daily_conversions = daily_visitors * (conversion_rate + noise)
    return daily_conversions.astype(int)


def get_conversions_array(daily_visitors, daily_conversions):
    n_visitors = sum(daily_visitors)
    n_converted = sum(daily_conversions)
    converted_array = np.ones(n_converted)
    non_converted_array = np.zeros(n_visitors - n_converted)
    return np.concatenate([converted_array, non_converted_array])


def grid_configs(dictionary):
    for combination in product(*dictionary.values()):
        yield dict(zip(dictionary.keys(), combination))
