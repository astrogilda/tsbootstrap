import os
import inspect
from typing import Optional, List
import logging


class LogConfigurator:
    """
    A class used to encapsulate logger configuration

    Attributes
    ----------
    name : str
        name of the logger.
    level : str
        logging level.
    handlers : list
        list of handlers to be added to the logger.
    formatter : logging.Formatter
        default formatter for the logger.

    Methods
    -------
    get_logger():
        Returns a configured logger.
    add_handler(handler: logging.Handler):
        Adds a handler to the logger.
    remove_handler(handler_type: type):
        Removes a handler from the logger based on its type.
    set_level(level: str):
        Sets the logging level.
    add_filter(filter_: logging.Filter):
        Adds a filter to all handlers of the logger.
    set_formatter(format_: str):
        Sets a formatter for all handlers of the logger.
    add_file_handler(filename: Optional[str] = None):
        Adds a file handler to the logger. If no filename is provided, 
        uses the calling module's name as the filename.

    Examples
    --------
    >>> from utils.logger import LogConfigurator

    >>> logger = LogConfigurator(name="my_logger", level="INFO")
    >>> logger.add_file_handler()
    >>> logger.get_logger().info("Hello World!")
    # The log message will be written to 'logs/mymodule_logger.log'
    """

    def __init__(self, name: str, level: Optional[str] = "INFO", handlers: Optional[List[logging.Handler]] = None, format_: Optional[str] = '%(asctime)s - %(name)s - %(levelname)s - %(message)s') -> None:
        self.name = name
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.handlers = handlers if handlers else [logging.StreamHandler()]
        self.formatter = logging.Formatter(format_)

        self._logger = logging.getLogger(self.name)
        self._logger.setLevel(self.level)

        # Apply all handlers
        for handler in self.handlers:
            self.add_handler(handler)

    def get_logger(self) -> logging.Logger:
        """
        Returns the configured logger.

        Returns
        -------
        logging.Logger
            Configured logger.
        """
        return self._logger

    def add_handler(self, handler: logging.Handler) -> None:
        """
        Adds a handler to the logger.

        Parameters
        ----------
        handler : logging.Handler
            A logging handler instance.
        """
        handler.setLevel(self.level)

        # Set formatter
        handler.setFormatter(self.formatter)

        # Add the handler to the logger
        self._logger.addHandler(handler)

    def remove_handler(self, handler_type: type) -> None:
        """
        Removes a handler from the logger based on its type.

        Parameters
        ----------
        handler_type : type
            The type of the handler to remove.
        """
        for handler in self._logger.handlers:
            if isinstance(handler, handler_type):
                self._logger.removeHandler(handler)

    def set_level(self, level: str) -> None:
        """
        Sets the logging level.

        Parameters
        ----------
        level : str
            The level to set for logging.
        """
        self.level = getattr(logging, level.upper(), logging.INFO)
        self._logger.setLevel(self.level)
        for handler in self._logger.handlers:
            handler.setLevel(self.level)

    def add_filter(self, filter_: logging.Filter) -> None:
        """
        Adds a filter to all handlers of the logger.

        Parameters
        ----------
        filter_ : logging.Filter
            A logging filter instance.
        """
        for handler in self._logger.handlers:
            handler.addFilter(filter_)

    def set_formatter(self, format_: str) -> None:
        """
        Sets a formatter for all handlers of the logger.

        Parameters
        ----------
        format_ : str
            A string representing the format to use for the formatter.
        """
        self.formatter = logging.Formatter(format_)
        for handler in self._logger.handlers:
            handler.setFormatter(self.formatter)

    def add_file_handler(self, filename: Optional[str] = None) -> None:
        """
        Adds a file handler to the logger. If no filename is provided, 
        uses the calling module's name as the filename.

        Parameters
        ----------
        filename : Optional[str], optional
            The filename to use for the log file. If not provided, 
            the calling module's name will be used, by default None.
        """
        # If no filename is provided, use the calling module's name as the filename
        if filename is None:
            frame = inspect.stack()[1]
            module = inspect.getmodule(frame[0])
            filename = os.path.join(
                'logs', f"{os.path.basename(module.__file__).split('.')[0]}_logger.log")

        file_handler = logging.FileHandler(filename)
        self.add_handler(file_handler)
