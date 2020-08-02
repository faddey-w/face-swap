from reface.config import Config


def make_compatible_config(cfg: Config):
    _default(cfg, "TEST.TEST_PERIOD", cfg.TRAINING.CHECKPOINT_PERIOD)
    _default(cfg, "TEST.N_TEST_BATCHES", 100)
    _default(cfg, "INPUT.MIN_FACE_SIZE", None)
    _default(cfg, "GENERATOR.AAD_NORM", "instance")
    _default(cfg, "GENERATOR.AAD_NORM", "instance")
    _replace(
        cfg, "GENERATOR.AAD_USE_ADAPTIVE_NORM", True, "GENERATOR.AAD_NORM", "AdaIN"
    )
    _default(cfg, "TRAINING.CHECKPOINTS_KEEP_PERIOD", None)
    _default(cfg, "TRAINING.OPT_STEP_PERIOD", 1)
    _default(cfg, "TRAINING.INIT_CHECKPOINT", None)


def _get_subcfg(cfg, path):
    parts = path.split(".")
    for p in parts[:-1]:
        cfg = getattr(cfg, p)
    return cfg, parts[-1]


def _default(cfg, path, default):
    cfg, field = _get_subcfg(cfg, path)
    if not hasattr(cfg, field):
        print(f"set default cfg.{path} = {default!r}")
        setattr(cfg, field, default)


def _replace(cfg, old_path, old_value, new_path, new_value):
    try:
        old_sub, old_field = _get_subcfg(cfg, old_path)
        need_replace = getattr(old_sub, old_field) == old_value
    except AttributeError:
        need_replace = False
    if need_replace:
        print(
            f"deprecated cfg.{old_path} = {old_value!r}  ->  "
            f"cfg.{new_path} = {new_value!r}"
        )
        new_sub, new_field = _get_subcfg(cfg, new_path)
        setattr(new_sub, new_field, new_value)
