"""
Utility functions for filtering configuration files.
These are commonly used to find experiment paths that match certain criteria.
"""

from abc import ABC, abstractmethod

from config.config import Config


class ConfigFilter(ABC):
    """
    A filter for configuration files.
    """

    @abstractmethod
    def __call__(self, cfg: Config) -> bool:
        """
        Check if the configuration file passes the filter.

        :param cfg: The configuration file to check.
        :return: True if the configuration file passes the filter, False otherwise.
        """
        pass


class DryRunFilter(ConfigFilter):
    """
    A filter that checks if the configuration file has dry run enabled.
    """

    def __init__(self, dry_run: bool) -> None:
        self.dry_run = dry_run

    def __call__(self, cfg: Config) -> bool:
        return cfg.execution.dry_run == self.dry_run


class ModelFilter(ConfigFilter):
    """
    A filter that checks if the configuration file has a specific model.
    """

    def __init__(self, model_cfg: dict) -> None:
        self.model_cfg = model_cfg

    def __call__(self, cfg: Config) -> bool:
        return cfg.model == self.model_cfg
