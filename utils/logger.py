import logging
import sys


class Logger:
    def __init__(self, log_file=None):
        self.logger = logging.getLogger("AIC4Tracker")
        self.logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        self.logger.addHandler(ch)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)

    def info(self, msg):    self.logger.info(msg)
    def warning(self, msg): self.logger.warning(msg)
    def error(self, msg):   self.logger.error(msg)
    def debug(self, msg):   self.logger.debug(msg)