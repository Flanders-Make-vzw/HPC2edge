class Config:
    def __init__(self, dictionary: dict):
        self._dict = dictionary
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)


def log_scalars(writer, scalars, section, step):
    for s in scalars:
        writer.add_scalar(f"{section}/{s}", scalars[s], global_step=step)
