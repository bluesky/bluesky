import epics
import asyncio


class PVSuspender:
    """
    A class to manage the callback interface between asyincio and
    pyepics.

    """
    def __init__(self, RE, pv_name, *, sleep=0, loop=None):
        """
        Parameters
        ----------
        RE : RunEngine
            The run engine instance this should work on

        pv_name : str
            The PV to watch for changes to determine if the
            scan should be suspended

        sleep : float, optional
            How long to wait in seconds after the resume condition is met
            before marking the event as done.  Defaults to 0

        loop : BaseEventLoop, optional
            The event loop to work on
        """
        if loop is None:
            loop = asyncio.get_event_loop()
        self._loop = loop
        self.RE = RE
        self._ev = None
        self._sleep = sleep

        self._pv = epics.PV(pv_name, auto_monitor=True)
        self._pv.add_callback(self)

    def _should_suspend(self, value):
        """
        Determine if the current value of the PV is such
        that we need to tell the scan to suspend

        Parameters
        ----------
        value : object
            The value to evaluate to determine if we should
            suspend

        Returns
        -------
        suspend : bool
            True means suspend
        """
        raise NotImplementedError()

    def _should_resume(self, value):
        """
        Determine if the scan is ready to automatically
        restart.

        Parameters
        ----------
        value : object
            The value to evaluate to determine if we should
            resume

        Returns
        -------
        suspend : bool
            True means resume
        """
        raise NotImplementedError()

    def __call__(self, **kwargs):
        """
        Make the class callable so that we can
        pass it off to the pyepics callback stack.

        This expects the massive blob that comes from pyepics
        """
        value = kwargs['value']
        # TODO does this need thread locking? Depends on if
        # more than one thread can service the same PV at the
        # pyepics layer.
        if self._ev is None:
            # in the case where either have never been
            # called or have already fully cycled once
            if self._should_suspend(value):
                self._ev = asyncio.Event(loop=self._loop)

                self._loop.call_soon_threadsafe(
                    self.RE.request_suspend,
                    self._ev.wait(),
                    self._sleep)
        else:
            if self._should_resume(value):
                def local():
                    self._loop.call_later(self._sleep, self._ev.set)
                self._loop.call_soon_threadsafe(local)
                # clear that we have an event
                self._ev = None


class PVSuspendFloor(PVSuspender):
    def __init__(self, RE, pv, suspend_thresh, *,
                 resume_thresh=None, **kwargs):
        super().__init__(RE, pv, **kwargs)
        self._suspend_thresh = suspend_thresh
        if resume_thresh is None:
            resume_thresh = suspend_thresh
        self._resume_thresh = resume_thresh
        if resume_thresh < suspend_thresh:
            raise ValueError("Resume threshold must be equal or greater "
                             "than suspend threshold, you passed: "
                             "suspend: {}  resume: {}".format(
                                 suspend_thresh, resume_thresh))

    def _should_resume(self, value):
        return value > self._resume_thresh

    def _should_suspend(self, value):
        return value < self._suspend_thresh


class PVSuspendCeil(PVSuspender):
    def __init__(self, RE, pv, suspend_thresh, *,
                 resume_thresh=None, **kwargs):
        super().__init__(RE, pv, **kwargs)
        self._suspend_thresh = suspend_thresh
        if resume_thresh is None:
            resume_thresh = suspend_thresh
        self._resume_thresh = resume_thresh
        if resume_thresh > suspend_thresh:
            raise ValueError("Resume threshold must be equal or less "
                             "than suspend threshold, you passed: "
                             "suspend: {}  resume: {}".format(
                                 suspend_thresh, resume_thresh))

    def _should_resume(self, value):
        return value < self._resume_thresh

    def _should_suspend(self, value):
        return value > self._suspend_thresh


class PVSuspendBoolHigh(PVSuspender):

    def _should_suspend(self, value):
        return bool(value)

    def _should_resume(self, value):
        return not bool(value)


class PVSuspendBoolLow(PVSuspender):

    def _should_suspend(self, value):
        return not bool(value)

    def _should_resume(self, value):
        return bool(value)
