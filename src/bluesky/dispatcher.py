"""Dispatch documents to the callbacks that consume them."""

import typing
from itertools import count
from warnings import warn

from event_model import DocumentNames

from .utils import CallbackRegistry

__all__ = ["Dispatcher", "DocumentNames"]


class Dispatcher:
    """Dispatch documents to user-defined consumers on the main thread.

    Dispatchers chain: a document goes to the ``parent``'s subscribers first,
    then to this one's.
    """

    def __init__(self, parent: "Dispatcher | None" = None, *, ignore_exceptions: bool = False) -> None:
        self._parent = parent
        self.cb_registry = CallbackRegistry(allowed_sigs=DocumentNames, ignore_exceptions=ignore_exceptions)
        # Shared along the chain, so a session's token never names one of a plan's subscribers.
        self._counter: typing.Iterator[int] = parent._counter if parent is not None else count()
        # public token -> the registry tokens it stands for
        self._token_mapping: dict[int, list[typing.Any]] = {}

    def process(self, name: DocumentNames, doc) -> None:
        """
        Dispatch document ``doc`` of type ``name`` to the callback registry.

        Parameters
        ----------
        name : {'start', 'descriptor', 'event', 'stop'}
        doc : dict
        """
        if self._parent is not None:
            self._parent.process(name, doc)
            # Read live, so changing `RE.ignore_callback_exceptions` reaches a running plan.
            self.cb_registry.ignore_exceptions = self._parent.ignore_exceptions
        exceptions = self.cb_registry.process(name, name.name, doc)
        for exc, traceback in exceptions:  # noqa: B007
            warn(  # noqa: B028
                "A %r was raised during the processing of a %s "  # noqa: UP031
                "Document. The error will be ignored to avoid "
                "interrupting data collection. To investigate, "
                "set RunEngine.ignore_callback_exceptions = False "
                "and run again." % (exc, name.name)
            )

    def subscribe(self, func, name="all") -> int:
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

    def unsubscribe(self, token: int) -> None:
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

    def unsubscribe_all(self) -> None:
        """Unregister all callbacks from the dispatcher."""
        for public_token in list(self._token_mapping.keys()):
            self.unsubscribe(public_token)

    @property
    def ignore_exceptions(self) -> bool:
        """Whether a raising subscriber is warned about rather than raised. Shared along the chain."""
        if self._parent is not None:
            return self._parent.ignore_exceptions
        return self.cb_registry.ignore_exceptions

    @ignore_exceptions.setter
    def ignore_exceptions(self, val: bool) -> None:
        if self._parent is not None:
            self._parent.ignore_exceptions = val
        else:
            self.cb_registry.ignore_exceptions = val
