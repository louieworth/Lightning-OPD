"""Resolve per-epoch YAML budgets once, including saved normalized configs."""


def normalize_epochs(cfg):
    cfg = dict(cfg)
    epoch = cfg.get("epoch", 1)
    steps = cfg.get("num_rollout_per_epoch", cfg["num_rollout"])
    if type(epoch) is not int or epoch < 1:
        raise ValueError("epoch must be a positive integer.")
    if type(steps) is not int or steps < 1:
        raise ValueError("num_rollout per epoch must be a positive integer.")
    cfg.update(epoch=epoch, num_rollout_per_epoch=steps, num_rollout=steps * epoch)
    return cfg
