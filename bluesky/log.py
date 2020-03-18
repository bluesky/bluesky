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

__all__ = ('config_bluesky_logging', 'get_handler',
           'LogFormatter', 'set_handler')


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


class ComposableLogAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        # The logging.LoggerAdapter siliently ignores `extra` in this usage:
        # log_adapter.debug(msg, extra={...})
        # and passes through log_adapater.extra instead. This subclass merges
        # the extra passed via keyword argument with the extra in the
        # attribute, giving precedence to the keyword argument.
        kwargs["extra"] = {**self.extra, **kwargs.get('extra', {})}
        return msg, kwargs


class LogFormatter(logging.Formatter):
    """Log formatter for bluesky records.

    Adapted from the log formatter used in Tornado.
    Key features of this formatter are:

    * Color support when logging to a terminal that supports it.
    * Timestamps on every log line.
    * Includes extra record attributes (old_state, new_state, msg_command,
      doc_name, doc_uid) when present.

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
        message = []
        message.append(record.getMessage())
        record.message = ' '.join(message)
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


plain_log_format = "[%(levelname)1.1s %(asctime)s.%(msecs)03d %(module)15s:%(lineno)5d] %(message)s"
color_log_format = ("%(color)s[%(levelname)1.1s %(asctime)s.%(msecs)03d "
                    "%(module)15s:%(lineno)5d]%(end_color)s %(message)s")


logger = logging.getLogger('bluesky')
doc_logger = logging.getLogger('bluesky.emit_document')
msg_logger = logging.getLogger('bluesky.RE.msg')
state_logger = logging.getLogger('bluesky.RE.state')
current_handler = None


def validate_level(level) -> int:
    '''
    Return a int for level comparison

    '''
    if isinstance(level, int):
        levelno = level
    elif isinstance(level, str):
        levelno = logging.getLevelName(level)

    if isinstance(levelno, int):
        return levelno
    else:
        raise ValueError("Your level is illegal, please use one of python logging string")


def _set_handler_with_logger(logger_name='bluesky', file=sys.stdout, datefmt='%H:%M:%S', color=True,
                             level='WARNING'):
    if isinstance(file, str):
        handler = logging.FileHandler(file)
    else:
        handler = logging.StreamHandler(file)
    levelno = validate_level(level)
    handler.setLevel(levelno)
    if color:
        format = color_log_format
    else:
        format = plain_log_format
    handler.setFormatter(
        LogFormatter(format, datefmt=datefmt))
    logging.getLogger(logger_name).addHandler(handler)
    if logger.getEffectiveLevel() > levelno:
        logger.setLevel(levelno)


def config_bluesky_logging(file=sys.stdout, datefmt='%H:%M:%S', color=True, level='WARNING'):
    """
    Set a new handler on the ``logging.getLogger('bluesky')`` logger.

    If this is called more than once, the handler from the previous invocation
    is removed (if still present) and replaced.

    Parameters
    ----------
    file : object with ``write`` method or filename string
        Default is ``sys.stdout``.
    datefmt : string
        Date format. Default is ``'%H:%M:%S'``.
    color : boolean
        Use ANSI color codes. True by default.
    level : str or int
        Python logging level, given as string or corresponding integer.
        Default is 'WARNING'.

    Returns
    -------
    handler : logging.Handler
        The handler, which has already been added to the 'bluesky' logger.

    Examples
    --------
    Log to a file.

    >>> config_bluesky_logging(file='/tmp/what_is_happening.txt')

    Include the date along with the time. (The log messages will always include
    microseconds, which are configured separately, not as part of 'datefmt'.)

    >>> config_bluesky_logging(datefmt="%Y-%m-%d %H:%M:%S")

    Turn off ANSI color codes.

    >>> config_bluesky_logging(color=False)

    Increase verbosity: show level INFO or higher.

    >>> config_bluesky_logging(level='INFO')
    """
    global current_handler
    if isinstance(file, str):
        handler = logging.FileHandler(file)
    else:
        handler = logging.StreamHandler(file)
    levelno = validate_level(level)
    handler.setLevel(levelno)
    if color:
        format = color_log_format
    else:
        format = plain_log_format
    handler.setFormatter(
        LogFormatter(format, datefmt=datefmt))
    if current_handler in logger.handlers:
        logger.removeHandler(current_handler)
    logger.addHandler(handler)
    current_handler = handler
    if logger.getEffectiveLevel() > levelno:
        logger.setLevel(levelno)
    return handler


set_handler = config_bluesky_logging  # for back-compat


def get_handler():
    """
    Return the handler configured by the most recent call to :func:`config_bluesky_logging`.

    If :func:`config_bluesky_logging` has not yet been called, this returns ``None``.
    """
    return current_handler
