from reface.config import Config


def make_compatible_config(cfg: Config):
    _default(cfg, "TEST.TEST_PERIOD", cfg.TRAINING.CHECKPOINT_PERIOD)
    _default(cfg, "TEST.N_TEST_BATCHES", 100)
    _default(cfg, "INPUT.MIN_FACE_SIZE", None)
    _default(cfg, "GENERATOR.AAD_NORM", "instance")
    if getattr(cfg.GENERATOR, "AAD_USE_ADAPTIVE_NORM", False):
        print(
            "deprecated cfg.GENERATOR.AAD_USE_ADAPTIVE_NORM = True  ->  "
            "cfg.GENERATOR.AAD_NORM = 'AdaIN"
        )
        cfg.GENERATOR.AAD_NORM = "AdaIN"


def _default(cfg, path, default):
    parts = path.split(".")
    for p in parts[:-1]:
        cfg = getattr(cfg, p)
    if not hasattr(cfg, parts[-1]):
        print(f"set default cfg.{path} = {default!r}")
        setattr(cfg, parts[-1], default)
