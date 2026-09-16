"""Dispatch documents to the callbacks that consume them."""

from itertools import count
from warnings import warn

from event_model import DocumentNames

from .utils import CallbackRegistry

__all__ = ["Dispatcher", "DocumentNames"]


class Dispatcher:
    """Dispatch documents to user-defined consumers on the main thread."""

    def __init__(self):
        self.cb_registry = CallbackRegistry(allowed_sigs=DocumentNames)
        self._counter = count()
        self._token_mapping = dict()  # noqa: C408

    def process(self, name, doc):
        """
        Dispatch document ``doc`` of type ``name`` to the callback registry.

        Parameters
        ----------
        name : {'start', 'descriptor', 'event', 'stop'}
        doc : dict
        """
        exceptions = self.cb_registry.process(name, name.name, doc)
        for exc, traceback in exceptions:  # noqa: B007
            warn(  # noqa: B028
                "A %r was raised during the processing of a %s "  # noqa: UP031
                "Document. The error will be ignored to avoid "
                "interrupting data collection. To investigate, "
                "set RunEngine.ignore_callback_exceptions = False "
                "and run again." % (exc, name.name)
            )

    def subscribe(self, func, name="all"):
        """
        Register a callback function to consume documents.

        .. versionchanged :: 0.10.0
            The order of the arguments was swapped and the ``name``
            argument has been given a default value, ``'all'``. Because the
            meaning of the arguments is unambiguous (they must be a callable
            and a string, respectively) the old order will be supported
            indefinitely, with a warning.

        .. versionchanged :: 0.10.0
            The order of the arguments was swapped and the ``name``
            argument has been given a default value, ``'all'``. Because the
            meaning of the arguments is unambiguous (they must be a callable
            and a string, respectively) the old order will be supported
            indefinitely, with a warning.

        Parameters
        ----------
        func: callable
            expecting signature like ``f(name, document)``
            where name is a string and document is a dict
        name : {'all', 'start', 'descriptor', 'event', 'stop'}, optional
            the type of document this function should receive ('all' by
            default).

        Returns
        -------
        token : int
            an integer ID that can be used to unsubscribe

        See Also
        --------
        :meth:`Dispatcher.unsubscribe`
            an integer token that can be used to unsubscribe
        """
        if callable(name) and isinstance(func, str):
            name, func = func, name
            warn(  # noqa: B028
                "The order of the arguments has been changed. Because the "
                "meaning of the arguments is unambiguous, the old usage will "
                "continue to work indefinitely, but the new usage is "
                "encouraged: call subscribe(func, name) instead of "
                "subscribe(name, func). Additionally, the 'name' argument "
                "has become optional. Its default value is 'all'."
            )
        if name == "all":
            private_tokens = []
            for key in DocumentNames:
                private_tokens.append(self.cb_registry.connect(key, func))
            public_token = next(self._counter)
            self._token_mapping[public_token] = private_tokens
            return public_token

        name = DocumentNames[name]
        private_token = self.cb_registry.connect(name, func)
        public_token = next(self._counter)
        self._token_mapping[public_token] = [private_token]
        return public_token

    def unsubscribe(self, token):
        """
        Unregister a callback function using its integer ID.

        Parameters
        ----------
        token : int
            the integer ID issued by :meth:`Dispatcher.subscribe`

        See Also
        --------
        :meth:`Dispatcher.subscribe`
        """
        for private_token in self._token_mapping.pop(token, []):
            self.cb_registry.disconnect(private_token)

    def unsubscribe_all(self):
        """Unregister all callbacks from the dispatcher."""
        for public_token in list(self._token_mapping.keys()):
            self.unsubscribe(public_token)

    @property
    def ignore_exceptions(self):
        return self.cb_registry.ignore_exceptions

    @ignore_exceptions.setter
    def ignore_exceptions(self, val):
        self.cb_registry.ignore_exceptions = val
