import asyncio
from datetime import datetime, timedelta
from abc import ABCMeta, abstractmethod, abstractproperty
import operator
import threading
from functools import partial
from warnings import warn


class SuspenderBase(metaclass=ABCMeta):
    """An ABC to manage the callbacks between asyincio and pyepics.


    Parameters
    ----------
    signal : `ophyd.Signal`
        The signal to watch for changes to determine if the
        scan should be suspended

    sleep : float, optional
        How long to wait in seconds after the resume condition is met
        before marking the event as done.  Defaults to 0

    pre_plan : iterable or iterator or generator function, optional
            a generator, list, or similar containing `Msg` objects

    post_plan : iterable or iterator or generator function, optional
            a generator, list, or similar containing `Msg` objects

    tripped_message : str, optional
        Message to include in the trip notification
    """
    def __init__(self, signal, *, sleep=0, pre_plan=None, post_plan=None,
                 tripped_message=''):
        """
        """
        self.RE = None
        self._ev = None
        self._tripped = False
        self._tripped_message = tripped_message
        self._sleep = sleep
        self._lock = threading.Lock()
        self._sig = signal
        self._pre_plan = pre_plan
        self._post_plan = post_plan

    def __repr__(self):
        return (
            "{}({!r}, sleep={}, pre_plan={}, post_plan={},"
            "tripped_message={})".format(
                type(self).__name__,
                self._sig,
                self._sleep,
                self._pre_plan,
                self._post_plan,
                self._tripped_message,
            )
        )

    def install(self, RE, *, event_type=None):
        """Install callback on signal

        This (re)installs the required callbacks at the pyepics level

        Parameters
        ----------

        RE : RunEngine
            The run engine instance this should work on

        event_type : str, optional
            The event type (subscription type) to watch
        """
        with self._lock:
            self.RE = RE
        self._sig.subscribe(self, event_type=event_type, run=True)

    def remove(self):
        """Disable the suspender

        Removes the callback at the pyepics level
        """
        self._sig.clear_sub(self)
        with self._lock:
            if self.RE is not None:
                self.__set_event(self.RE._loop)
            self.RE = None
            self._tripped = False

    @abstractmethod
    def _should_suspend(self, value):
        """
        Determine if the current value of the signal is such
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
        """Make the class callable so that we can
        pass it off to the ophyd callback stack.

        This expects the massive blob that comes from ophyd
        """
        with self._lock:
            if self.RE is None:
                return
            loop = self.RE._loop

            if self._should_suspend(value):
                self._tripped = True
                # this does dirty things with internal state
                if self._ev is None and self.RE is not None:
                    self.__make_event()
                    if self._ev is None:
                        raise RuntimeError("Could not create the ")
                    cb = partial(
                        self.RE.request_suspend,
                        self._ev.wait,
                        pre_plan=self._pre_plan,
                        post_plan=self._post_plan,
                        justification=self._get_justification(),
                    )
                    if self.RE.state.is_running:
                        loop.call_soon_threadsafe(cb)
            elif self._should_resume(value):
                self.__set_event(loop)
                self._tripped = False

    def __make_event(self):
        """Make or return the asyncio.Event to use as a bridge."""
        assert self._lock.locked()
        if self._ev is None and self.RE is not None:
            th_ev = threading.Event()

            def really_make_the_event():
                self._ev = asyncio.Event()
                th_ev.set()

            h = self.RE._loop.call_soon_threadsafe(really_make_the_event)
            if not th_ev.wait(0.1):
                h.cancel()
        return self._ev

    def __set_event(self, loop):
        """Notify the event that it can resume"""
        assert self._lock.locked()
        if self._ev:
            ev = self._ev
            sleep = self._sleep

            def local():
                ts = (datetime.now() + timedelta(seconds=sleep)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                print(
                    "Suspender {!r} reports a return to nominal "
                    "conditions. Will sleep for {} seconds and then "
                    "release suspension at {}.".format(self, sleep, ts)
                )
                # we can use call_later here because this function
                # is scheduled to be run in the event loop thread
                # by the `call_soon_threadsafe` call just below.
                loop.call_later(sleep, ev.set)

            loop.call_soon_threadsafe(local)
        # clear that we have an event
        self._ev = None

    def get_futures(self):
        """Return a list of futures to wait on.

        This will only work correctly if this suspender is 'installed'
        and watching a signal

        Returns
        -------
        futs : list
            List of futures to wait on

        justification : str
            String explaining why the suspender is tripped
        """
        if not self.tripped:
            return [], ""
        with self._lock:
            return [self.__make_event().wait], self._get_justification()

    @property
    def tripped(self):
        return self._tripped

    def _get_justification(self):
        if not self.tripped:
            return ''

        template = 'Suspender of type {} stopped by signal {!r}'
        just = template.format(self.__class__.__name__, self._sig)
        return ': '.join(s for s in (just, self._tripped_message)
                         if s)


class SuspendBoolHigh(SuspenderBase):
    """
    Suspend when a boolean signal goes high; resume when it goes low.

    Parameters
    ----------
    signal : `ophyd.Signal`
        The signal to watch for changes to determine if the
        scan should be suspended

    sleep : float, optional
        How long to wait in seconds after the resume condition is met
        before marking the event as done.  Defaults to 0

    pre_plan : iterable or iterator, optional
            a generator, list, or similar containing `Msg` objects

    post_plan : iterable or iterator, optional
            a generator, list, or similar containing `Msg` objects
    """

    def _should_suspend(self, value):
        return bool(value)

    def _should_resume(self, value):
        return not bool(value)

    def _get_justification(self):
        if not self.tripped:
            return ''

        just = 'Signal {} is high'.format(self._sig.name)
        return ': '.join(s for s in (just, self._tripped_message)
                         if s)


class SuspendBoolLow(SuspenderBase):
    """
    Suspend when a boolean signal goes low; resume when it goes high.

    Parameters
    ----------
    signal : `ophyd.Signal`
        The signal to watch for changes to determine if the
        scan should be suspended

    sleep : float, optional
        How long to wait in seconds after the resume condition is met
        before marking the event as done.  Defaults to 0

    pre_plan : iterable or iterator, optional
            a generator, list, or similar containing `Msg` objects

    post_plan : iterable or iterator, optional
            a generator, list, or similar containing `Msg` objects
    """

    def _should_suspend(self, value):
        return not bool(value)

    def _should_resume(self, value):
        return bool(value)

    def _get_justification(self):
        if not self.tripped:
            return ''

        just = 'Signal {} is low'.format(self._sig.name)
        return ': '.join(s for s in (just, self._tripped_message)
                         if s)


class _Threshold(SuspenderBase):
    """
    Private base class for suspenders that watch when a scalar
    signal fall above or below a threshold.  Allow for a possibly different
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


class SuspendFloor(_Threshold):
    """
    Suspend when a scalar falls below a threshold.

    Optionally, the threshold to resume can be set to be greater than the
    threshold to suspend.

    Parameters
    ----------
    signal : `ophyd.Signal`
        The signal to watch for changes to determine if the
        scan should be suspended

    suspend_thresh : float
        Suspend if the signal value falls below this value

    resume_thresh : float, optional
        Resume when the signal value rises above this value.  If not
        given set to `suspend_thresh`.  Must be greater than `suspend_thresh`.

    sleep : float, optional
        How long to wait in seconds after the resume condition is met
        before marking the event as done.  Defaults to 0

    pre_plan : iterable or iterator, optional
            a generator, list, or similar containing `Msg` objects

    post_plan : iterable or iterator, optional
            a generator, list, or similar containing `Msg` objects
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

    def _get_justification(self):
        if not self.tripped:
            return ''

        just = ('Signal {} = {!r} is below {}'
                ''.format(self._sig.name, self._sig.get(),
                          self._suspend_thresh)
                )
        return ': '.join(s for s in (just, self._tripped_message)
                         if s)


class SuspendCeil(_Threshold):
    """
    Suspend when a scalar rises above a threshold.

    Optionally, the threshold to resume can be set to be less than the
    threshold to suspend.

    Parameters
    ----------
    signal : `ophyd.Signal`
        The signal to watch for changes to determine if the
        scan should be suspended

    suspend_thresh : float
        Suspend if the signal value falls below this value

    resume_thresh : float, optional
        Resume when the signal value rises above this value.  If not
        given set to `suspend_thresh`.  Must be greater than `suspend_thresh`.

    sleep : float, optional
        How long to wait in seconds after the resume condition is met
        before marking the event as done.  Defaults to 0

    pre_plan : iterable or iterator, optional
            a generator, list, or similar containing `Msg` objects

    post_plan : iterable or iterator, optional
            a generator, list, or similar containing `Msg` objects
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

    def _get_justification(self):
        if not self.tripped:
            return ''

        just = ('Signal {} = {!r} is above {}'
                ''.format(self._sig.name, self._sig.get(),
                          self._suspend_thresh)
                )
        return ': '.join(s for s in (just, self._tripped_message)
                         if s)


class _SuspendBandBase(SuspenderBase):
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


class SuspendWhenOutsideBand(_SuspendBandBase):
    """
    Suspend when a scalar signal leaves a given band of values.

    Parameters
    ----------
    signal : `ophyd.Signal`
        The signal to watch for changes to determine if the
        scan should be suspended

    band_bottom, band_top : float
        The top and bottom of the band.  `band_top` must be
        strictly greater than `band_bottom`.

    sleep : float, optional
        How long to wait in seconds after the resume condition is met
        before marking the event as done.  Defaults to 0

    pre_plan : iterable or iterator, optional
            a generator, list, or similar containing `Msg` objects

    post_plan : iterable or iterator, optional
            a generator, list, or similar containing `Msg` objects
    """
    def _should_resume(self, value):
        return self._bot < value < self._top

    def _should_suspend(self, value):
        return not (self._bot < value < self._top)

    def _get_justification(self):
        if not self.tripped:
            return ''

        just = ('Signal {} = {!r} is outside of the range ({}, {})'
                ''.format(self._sig.name, self._sig.get(),
                          self._bot, self._top)
                )
        return ': '.join(s for s in (just, self._tripped_message)
                         if s)


class SuspendInBand(SuspendWhenOutsideBand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warn("SuspendInBand has been renamed SuspendWhenOutsideBand to make "
             "its meaning more clear. Its behavior has not changed.")


class SuspendOutBand(_SuspendBandBase):
    """
    Suspend when a scalar signal enters a given band of values.

    This is mostly here because it is the opposite of `SuspenderInBand`.

    Parameters
    ----------

    signal : `ophyd.Signal`
        The signal to watch for changes to determine if the
        scan should be suspended

    band_bottom, band_top : float
        The top and bottom of the band.  `band_top` must be
        strictly greater than `band_bottom`.

    sleep : float, optional
        How long to wait in seconds after the resume condition is met
        before marking the event as done.  Defaults to 0

    pre_plan : iterable or iterator, optional
            a generator, list, or similar containing `Msg` objects

    post_plan : iterable or iterator, optional
            a generator, list, or similar containing `Msg` objects
    """
    def __init__(self, *args, **kwargs):
        warn("bluesky.suspenders.SuspendOutBand is deprecated.")
        super().__init__(*args, **kwargs)

    def _should_resume(self, value):
        return not (self._bot < value < self._top)

    def _should_suspend(self, value):
        return (self._bot < value < self._top)

    def _get_justification(self):
        if not self.tripped:
            return ''

        just = ('Signal {} = {!r} is inside of the range ({}, {})'
                ''.format(self._sig.name, self._sig.get(),
                          self._bot, self._top)
                )
        return ': '.join(s for s in (just, self._tripped_message)
                         if s)


class SuspendWhenChanged(SuspenderBase):
    """
    Suspend when the monitored value deviates from the expected.

    Only resume if allowed AND when monitored equals expected.

    Notes
    -----

    This suspender is designed to require bluesky restart if value changes.

    USE CASE:

    :class:`~SuspendWhenChanged()` is useful when ``signal`` is an EPICS enumeration
    (`"mbbo" <https://wiki-ext.aps.anl.gov/epics/index.php/RRM_3-14_Multi-Bit_Binary_Output>`_)
    used with a multi-instrument facility.
    Choices predefined in the mbbo record are the
    names of instruments allowed to control any shared hardware.

    * The ``signal``, set by instrument staff outside of bluesky,
      names which instrument is allowed to control the hardware.
    * Other instruments not matching ``signal`` are expected **not** to
      control the hardware (they could use simulators instead or not operate
      the shared hardware).

    Since a decision of hardware *vs.* simulators is made at the
    time a bluesky session starts and ophyd objects are first created, the
    session needs to be aware immediately if the ``signal`` is changed.
    The default value of ``allow_resume=False`` defends this decision.
    If there is a mechanism engineered to toggle ophyd signals between
    hardware and simulators, one might consider ``allow_resume=True``.


    Parameters
    ----------

    signal : `ophyd.Signal`
        The signal to watch for changes to determine if the
        scan should be suspended

    expected_value : str, float, or int
        RunEngine operations will be suspended when signal deviates
        from this value.  If `None` (default), set to value of
        ``signal`` when object is created.

    allow_resume : bool
        Should RunEngine be allowed to resume once ``signal.value == expected``
        again?  Default value of ``False`` is expected for intended use case.

    sleep : float, optional
        How long to wait in seconds after the resume condition is met
        before marking the event as done.  Defaults to ``0``.

    pre_plan : iterable or callable, optional
       Plan to execute just before suspending. If callable, must
       take no arguments.

    post_plan : iterable or callable, optional
        Plan to execute just before resuming. If callable, must
        take no arguments.

    tripped_message : str, optional
        Message to include in the trip notification


    Examples
    --------

    .. code-block:: python

        # pause if this value changes in our session
        # note: this suspender is designed to require Bluesky restart if value changes
        suspend_instrument_in_use = SuspendWhenChanged(instrument_in_use)
        RE.install_suspender(suspend_instrument_in_use)

    Example EPICS database for APS 2-BM-A and 2-BM-B:

    .. code-block:: text

        record(mbbo, "2bm:instrument_in_use") {
            # instrument team sets this
            # For additional field names, see
            # https://epics.anl.gov/EpicsDocumentation/AppDevManuals/RecordRef/Recordref-25.html#HEADING25-15
            field(DESC, "instrument using beam now")
            field(ZRST, "none")
            field(ONST, "2-BM-A")
            field(TWST, "2-BM-B")
            # THST
            # FRST
            # FVST
            # ...
        }

    NOTE: **Always** make the zero choice (``ZRST``) in the mbbo record to be 'none'.
    This allows the instrument staff to designate that *no* instrument is allowed
    to control the shared hardware.  Start the names of the allowed instruments
    with ``ONST``.

    It is convenient for the multi-instrument facility to make this definition
    in EPICS rather than in a specific bluesky session.  The EPICS value could be
    useful in other contexts of instrument control beyond the realm of bluesky.
    """

    def __init__(self, signal, *,
                 expected_value=None,
                 allow_resume=False,
                 sleep=0, pre_plan=None, post_plan=None, tripped_message='',
                 **kwargs):

        self.expected_value = expected_value or signal.value
        self.allow_resume = allow_resume
        super().__init__(signal,
                         sleep=sleep,
                         pre_plan=pre_plan,
                         post_plan=post_plan,
                         tripped_message=tripped_message,
                         **kwargs)

    def _should_suspend(self, value):
        return value != self.expected_value

    def _should_resume(self, value):
        return self.allow_resume and value == self.expected_value

    def _get_justification(self):
        if not self.tripped:
            return ''

        just = (
            f'Signal {self._sig.name}'
            f', got "{self._sig.get()}"'
            f', expected "{self.expected_value}"'
            )
        if not self.allow_resume:
            just += '.  "RE.abort()" and then restart session to use new configuration.'
        return ': '.join(
            s
            for s in (just, self._tripped_message)
            if s)
