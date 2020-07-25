from reface.config import Config


def make_compatible_config(cfg: Config):
    _fill_field(cfg, "TEST.TEST_PERIOD", cfg.TRAINING.CHECKPOINT_PERIOD)
    _fill_field(cfg, "TEST.N_TEST_BATCHES", 100)
    _fill_field(cfg, "INPUT.MIN_FACE_SIZE", None)
    _fill_field(cfg, "GENERATOR.AAD_USE_ADAPTIVE_NORM", False)


def _fill_field(cfg, path, default):
    parts = path.split(".")
    for p in parts[:-1]:
        cfg = getattr(cfg, p)
    if not hasattr(cfg, parts[-1]):
        print(f"set default cfg.{path} = {default!r}")
        setattr(cfg, parts[-1], default)
