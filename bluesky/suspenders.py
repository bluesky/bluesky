import asyncio
from abc import ABCMeta, abstractmethod, abstractproperty
import operator
from threading import Lock


class PVSuspenderBase(metaclass=ABCMeta):
    """An ABC to manage the callbacks between asyincio and pyepics.


    Parameters
    ----------
    pv_name : str
        The PV to watch for changes to determine if the
        scan should be suspended

    sleep : float, optional
        How long to wait in seconds after the resume condition is met
        before marking the event as done.  Defaults to 0

    loop : BaseEventLoop, optional
        The event loop to work on

    """
    def __init__(self, signal, *, sleep=0, loop=None,
                 pre_plan=None, post_plan=None):
        """
        """
        if loop is None:
            loop = asyncio.get_event_loop()
        self.RE = None
        self._ev = None
        self._tripped = False
        self._sleep = sleep
        self._lock = Lock()
        self._sig = signal
        self._pre_plan = pre_plan
        self._post_plan = post_plan

    def install(self, RE, *, event_type=None):
        '''Install callback on PV

        This (re)installs the required callbacks at the pyepics level

        Parameters
        ----------

        RE : RunEngine
            The run engine instance this should work on

        event_type : str, optional
            The event type (subscription type) to watch
        '''
        self.RE = RE
        self._sig.subscribe(self, event_type=event_type, run=True)

    def remove(self):
        '''Disable the suspender

        Removes the callback at the pyepics level
        '''
        self.sig.clear_sub(self)
        self.RE = None

    @abstractmethod
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

    @abstractmethod
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

    def __call__(self, value, **kwargs):
        """
        Make the class callable so that we can
        pass it off to the pyepics callback stack.

        This expects the massive blob that comes from pyepics
        """

        with self._lock:
            if self._should_suspend(value):
                self._tripped = True
                if self._ev is None and self.RE is not None:
                    loop = self.RE.loop
                    self._ev = asyncio.Event(loop=loop)

                    loop.call_soon_threadsafe(
                        self.RE.request_suspend,
                        self._ev.wait(),
                        pre_plan=self._pre_plan,
                        post_plan=self._post_plan)
            elif self._should_resume(value):
                self._tripped = False
                if self._ev:

                    sleep = self._sleep
                    if self.RE is not None:
                        loop = self.RE.loop

                        def local():
                            loop.call_later(sleep, self._ev.set)
                        loop.call_soon_threadsafe(local)
                # clear that we have an event
                self._ev = None

    @property
    def tripped(self):
        return self._tripped


class PVSuspendBoolHigh(PVSuspenderBase):
    """
    Suspender which suspends the scan when a boolean PV
    goes high and resumes when the value goes low.

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
    def _should_suspend(self, value):
        return bool(value)

    def _should_resume(self, value):
        return not bool(value)


class PVSuspendBoolLow(PVSuspenderBase):
    """
    Suspender which suspends the scan when a boolean PV
    goes low and resumes when the value goes high.

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
    def _should_suspend(self, value):
        return not bool(value)

    def _should_resume(self, value):
        return bool(value)


class _PVThreshold(PVSuspenderBase):
    """
    Private base class for suspenders that watch when a scalar
    PV fall above or below a threshold.  Allow for a possibly different
    threshold to resume.
    """
    def __init__(self, signal, suspend_thresh, *,
                 resume_thresh=None, **kwargs):
        super().__init__(signal, **kwargs)
        self._suspend_thresh = suspend_thresh
        if resume_thresh is None:
            resume_thresh = suspend_thresh
        self._resume_thresh = resume_thresh
        self._validate()

    def _should_suspend(self, value):
        return self._op(value, self._suspend_thresh)

    def _should_resume(self, value):
        return not self._op(value, self._resume_thresh)

    @abstractproperty
    def _op(self):
        pass

    @abstractmethod
    def _validate(self):
        pass


class PVSuspendFloor(_PVThreshold):
    """
    A suspender that watches a scalar PV and suspends when it
    falls below a given threshold.  Optionally, the threshold to
    resume can be set to be greater than the threshold to suspend.

    Parameters
    ----------

    RE : RunEngine
        The run engine instance this should work on

    pv_name : str
        The PV to watch for changes to determine if the
        scan should be suspended

    suspend_thresh : float
        Suspend if the PV value falls below this value

    resume_thresh : float, optional
        Resume when the PV value rises above this value.  If not
        given set to `suspend_thresh`.  Must be greater than `suspend_thresh`.

    sleep : float, optional
        How long to wait in seconds after the resume condition is met
        before marking the event as done.  Defaults to 0

    loop : BaseEventLoop, optional
        The event loop to work on


    """
    def _validate(self):
        if self._resume_thresh < self._suspend_thresh:
            raise ValueError("Resume threshold must be equal or greater "
                             "than suspend threshold, you passed: "
                             "suspend: {}  resume: {}".format(
                                 self._suspend_thresh,
                                 self._resume_thresh))

    @property
    def _op(self):
        return operator.lt


class PVSuspendCeil(_PVThreshold):
    """
    A suspender that watches a scalar PV and suspends when it
    rises above a given threshold.  Optionally, the threshold to
    resume can be set to be less than the threshold to suspend.

    Parameters
    ----------

    RE : RunEngine
        The run engine instance this should work on

    pv_name : str
        The PV to watch for changes to determine if the
        scan should be suspended

    suspend_thresh : float
        Suspend if the PV value rises above this value

    resume_thresh : float, optional
        Resume when the PV value falls below this value.  If not
        given set to `suspend_thresh`.  Must be less than `suspend_thresh`.

    sleep : float, optional
        How long to wait in seconds after the resume condition is met
        before marking the event as done.  Defaults to 0

    loop : BaseEventLoop, optional
        The event loop to work on


    """
    def _validate(self):
        if self._resume_thresh > self._suspend_thresh:
            raise ValueError("Resume threshold must be equal or less "
                             "than suspend threshold, you passed: "
                             "suspend: {}  resume: {}".format(
                                 self._suspend_thresh,
                                 self._resume_thresh))

    @property
    def _op(self):
        return operator.gt


class _PVSuspendBandBase(PVSuspenderBase):
    """
    Private base-class for suspenders based on keeping a scalar inside
    or outside of a band
    """
    def __init__(self, signal, band_bottom, band_top, **kwargs):
        super().__init__(signal, **kwargs)
        if not band_bottom < band_top:
            raise ValueError("The bottom of the band must be strictly "
                             "less than the top of the band.\n"
                             "bottom: {}\ttop: {}".format(
                                 band_bottom, band_top)
                             )
        self._bot = band_bottom
        self._top = band_top


class PVSuspendInBand(_PVSuspendBandBase):
    """
    A suspender class to keep a scalar PV with in a band.  Suspends if
    the value leaves the band, resume when it re-enters.

    Parameters
    ----------

    RE : RunEngine
        The run engine instance this should work on

    pv_name : str
        The PV to watch for changes to determine if the
        scan should be suspended

    band_bottom, band_top : float
        The top and bottom of the band.  `band_top` must be
        strictly greater than `band_bottom`.

    sleep : float, optional
        How long to wait in seconds after the resume condition is met
        before marking the event as done.  Defaults to 0

    loop : BaseEventLoop, optional
        The event loop to work on

    """
    def _should_resume(self, value):
        return self._bot < value < self._top

    def _should_suspend(self, value):
        return not (self._bot < value < self._top)


class PVSuspendOutBand(_PVSuspendBandBase):
    """
    A suspender class to keep a scalar PV out of a band.  Suspends if
    the value enters the band and resumes when it leaves.

    This is mostly here because it is the opposite of `PVSuspenderInBand`.

    Parameters
    ----------

    RE : RunEngine
        The run engine instance this should work on

    pv_name : str
        The PV to watch for changes to determine if the
        scan should be suspended

    band_bottom, band_top : float
        The top and bottom of the band.  `band_top` must be
        strictly greater than `band_bottom`.

    sleep : float, optional
        How long to wait in seconds after the resume condition is met
        before marking the event as done.  Defaults to 0

    loop : BaseEventLoop, optional
        The event loop to work on

    """
    def _should_resume(self, value):
        return not (self._bot < value < self._top)

    def _should_suspend(self, value):
        return (self._bot < value < self._top)
