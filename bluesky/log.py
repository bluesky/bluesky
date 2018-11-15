# The LogFormatter is adapted light from tornado, which is licensed under
# Apache 2.0. See other_licenses/ in the repository directory.

import logging
import sys
try:
    import colorama
    colorama.init()
except ImportError:
    colorama = None
try:
    import curses
except ImportError:
    curses = None

__all__ = ('color_logs', 'plain_log_handler', 'color_log_handler')


def _stderr_supports_color():
    try:
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            if curses:
                curses.setupterm()
                if curses.tigetnum("colors") > 0:
                    return True
            elif colorama:
                if sys.stderr is getattr(colorama.initialise, 'wrapped_stderr',
                                         object()):
                    return True
    except Exception:
        # Very broad exception handling because it's always better to
        # fall back to non-colored logs than to break at startup.
        pass
    return False


class LogFormatter(logging.Formatter):
    """Log formatter used in Tornado, modified for Python3-only bluesky.
    Key features of this formatter are:
    * Color support when logging to a terminal that supports it.
    * Timestamps on every log line.
    * Robust against str/bytes encoding problems.
    This formatter is enabled automatically by
    `tornado.options.parse_command_line` or `tornado.options.parse_config_file`
    (unless ``--logging=none`` is used).
    Color support on Windows versions that do not support ANSI color codes is
    enabled by use of the colorama__ library. Applications that wish to use
    this must first initialize colorama with a call to ``colorama.init``.
    See the colorama documentation for details.
    __ https://pypi.python.org/pypi/colorama
    .. versionchanged:: 4.5
       Added support for ``colorama``. Changed the constructor
       signature to be compatible with `logging.config.dictConfig`.
    """
    DEFAULT_FORMAT = \
        '%(color)s[%(levelname)1.1s %(asctime)s %(module)s:%(lineno)d]%(end_color)s %(message)s'
    DEFAULT_DATE_FORMAT = '%y%m%d %H:%M:%S'
    DEFAULT_COLORS = {
        logging.DEBUG: 4,  # Blue
        logging.INFO: 2,  # Green
        logging.WARNING: 3,  # Yellow
        logging.ERROR: 1,  # Red
    }

    def __init__(self, fmt=DEFAULT_FORMAT, datefmt=DEFAULT_DATE_FORMAT,
                 style='%', color=True, colors=DEFAULT_COLORS):
        r"""
        :arg bool color: Enables color support.
        :arg str fmt: Log message format.
          It will be applied to the attributes dict of log records. The
          text between ``%(color)s`` and ``%(end_color)s`` will be colored
          depending on the level if color support is on.
        :arg dict colors: color mappings from logging level to terminal color
          code
        :arg str datefmt: Datetime format.
          Used for formatting ``(asctime)`` placeholder in ``prefix_fmt``.
        .. versionchanged:: 3.2
           Added ``fmt`` and ``datefmt`` arguments.
        """
        super().__init__(datefmt=datefmt)
        self._fmt = fmt

        self._colors = {}
        if color and _stderr_supports_color():
            if curses is not None:
                # The curses module has some str/bytes confusion in
                # python3.  Until version 3.2.3, most methods return
                # bytes, but only accept strings.  In addition, we want to
                # output these strings with the logging module, which
                # works with unicode strings.  The explicit calls to
                # unicode() below are harmless in python2 but will do the
                # right conversion in python 3.
                fg_color = (curses.tigetstr("setaf") or
                            curses.tigetstr("setf") or "")

                for levelno, code in colors.items():
                    self._colors[levelno] = str(curses.tparm(fg_color, code), "ascii")
                self._normal = str(curses.tigetstr("sgr0"), "ascii")
            else:
                # If curses is not present (currently we'll only get here for
                # colorama on windows), assume hard-coded ANSI color codes.
                for levelno, code in colors.items():
                    self._colors[levelno] = '\033[2;3%dm' % code
                self._normal = '\033[0m'
        else:
            self._normal = ''

    def format(self, record):
        record.message = record.getMessage()
        record.asctime = self.formatTime(record, self.datefmt)

        try:
            record.color = self._colors[record.levelno]
            record.end_color = self._normal
        except KeyError:
            record.color = ''
            record.end_color = ''

        formatted = self._fmt % record.__dict__

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            formatted = '{}\n{}'.format(formatted.rstrip(), record.exc_text)
        return formatted.replace("\n", "\n    ")


plain_log_format = "[%(levelname)1.1s %(asctime)s.%(msecs)03d %(module)s:%(lineno)d] %(message)s"
color_log_format = ("%(color)s[%(levelname)1.1s %(asctime)s.%(msecs)03d "
                    "%(module)s:%(lineno)d]%(end_color)s %(message)s")
log_date_format = "%H:%M:%S"
logger = logging.getLogger('bluesky')
color_log_handler = logging.StreamHandler(sys.stdout)
color_log_handler.setFormatter(
    LogFormatter(color_log_format, datefmt=log_date_format))
plain_log_handler = logging.StreamHandler(sys.stdout)
plain_log_handler.setFormatter(
    logging.Formatter(plain_log_format, datefmt=log_date_format))


def color_logs(color):
    """
    If True, add colorful logging handler and ensure plain one is removed.

    If False, do the opposite.
    """
    if color:
        to_remove, to_add = plain_log_handler, color_log_handler
    else:
        to_remove, to_add = color_log_handler, plain_log_handler
    if to_remove in logger.handlers:
        logger.removeHandler(to_remove)
    if to_add not in logger.handlers:
        logger.addHandler(to_add)


color_logs(True)
