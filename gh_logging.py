import logging
from pathlib import Path
import json

with open('config.json', 'r') as f:
    config = json.load(f)
dirs = config['dirs']
class CustomLogger:
    def __init__(self, name, log_folder, file_name, console_log=False):
        # Ensure the log folder exists
        self.log_folder = Path(log_folder)
        self.log_folder.mkdir(parents=True, exist_ok=True)

        # Combine the folder and file name to create the full path
        file_path = self.log_folder / file_name

        # Create a logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Default level; can be changed with set_level

        # Create a file handler and set level to debug
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.DEBUG)

        # Optionally create a console handler and set level to debug
        if console_log:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        # Create a formatter and set the formatter for the file handler
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Add the file handler to the logger
        self.logger.addHandler(file_handler)

    def set_level(self, level):
        """
        Sets the logging level for this logger and all its handlers.

        :param level: The logging level (e.g., logging.DEBUG, logging.INFO)
        """
        # Set the logger's level
        self.logger.setLevel(level)

        # Iterate over all handlers of the logger and set their level
        for handler in self.logger.handlers:
            handler.setLevel(level)
    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

# Example usage:
complog = CustomLogger('comps', dirs["02-Logs"])
complog.info('This is an info message for logger1.')

filelog = CustomLogger('files', dirs["02-Logs"])
filelog.error('This is an error message for filelog.')