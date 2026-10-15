"""OPD options shared by the CPU launcher, scorer and training workers."""

import math


LOSS_TOKEN_MODES = ("sample", "top_k_student", "top_k_teacher", "full_vocab")


def option(config, name, default):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def needs_teacher_features(config):
    return option(config, "opd_loss_token_mode", "sample") != "sample"


def validate_opd_options(config):
    mode = option(config, "opd_loss_token_mode", "sample")
    if mode not in LOSS_TOKEN_MODES:
        raise ValueError(f"opd_loss_token_mode must be one of {LOSS_TOKEN_MODES}.")
    for name, default in (("opd_token_top_k", 32), ("opd_distribution_chunk_size", 32)):
        value = option(config, name, default)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    maximum = option(config, "opd_loss_max_clip", 10.0)
    if maximum is not None and (type(maximum) not in (int, float) or not math.isfinite(maximum) or maximum <= 0):
        raise ValueError("opd_loss_max_clip must be finite and positive, or null.")
    minimum = option(config, "opd_log_prob_min_clip", -10.0)
    if minimum is not None and (type(minimum) not in (int, float) or not math.isfinite(minimum) or minimum >= 0):
        raise ValueError("opd_log_prob_min_clip must be finite and negative, or null.")
    if needs_teacher_features(config) and option(config, "context_parallel_size", 1) != 1:
        raise ValueError("OPD distribution loss currently requires context_parallel_size: 1.")
    if needs_teacher_features(config) and option(config, "train_backend", "megatron") != "megatron":
        raise ValueError("OPD distribution loss currently requires train_backend: megatron.")
