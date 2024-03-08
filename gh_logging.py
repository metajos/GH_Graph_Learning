import logging
from pathlib import Path


class LogManager:
    class __Logger:
        def __init__(self):
            self.loggers = {}

        def _new_logger(self, name, log_folder, file_name, console_log=False):
            log_folder = Path(log_folder)
            log_folder.mkdir(parents=True, exist_ok=True)

            file_path = log_folder / file_name

            logger = logging.getLogger(name)
            logger.setLevel(logging.DEBUG)

            file_handler = logging.FileHandler(file_path)
            file_handler.setLevel(logging.DEBUG)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            if console_log:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.DEBUG)
                console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
                console_handler.setFormatter(console_formatter)
                logger.addHandler(console_handler)

            self.loggers[name] = logger

        def get_logger(self, name):
            return self.loggers.get(name, None)

    instance = None

    def __new__(cls):
        if not cls.instance:
            cls.instance = cls.__Logger()
        return cls.instance

    @classmethod
    def get_logger(cls, name):
        instance = cls.__new__(cls)  # Ensure the singleton instance is created
        logger = instance.get_logger(name)
        if logger is not None:
            return logger
        else:
            raise AttributeError(f"Logger named '{name}' does not exist.")

    @classmethod
    def set_level_for_all(cls, level):
        instance = cls.__new__(cls)  # Ensure the singleton instance is created
        for logger in instance.loggers.values():
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(level)

    @classmethod
    def new_logger(cls, name, log_folder, file_name, console_log=False):
        instance = cls.__new__(cls)
        if name not in instance.loggers:
            instance._new_logger(name, log_folder, file_name, console_log)