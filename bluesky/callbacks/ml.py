"""Callbacks for AI/ML agents"""

from .core import CallbackBase, make_class_safe
from warnings import warn
import logging

logger = logging.getLogger(__name__)


@make_class_safe(logger=logger)
class AgentCallback(CallbackBase):
    """
    Callback for telling AI/ML agents about data, and generating reports.
    This callback does not support fully adaptive planning;
    however, can be used for streaming analysis, reports, or alerts.

    Parameters
    ----------
    agents :
        Each agent is required to have the methods ``tell`` and ``ask``
    independent_key : str
        Key for the independent variable (x) to send the agent. This will appear in doc["data"][independent_key].
    dependent_key : str
        Key for the dependent variable (y) to send the agent. This will appear in doc["data"][dependent_key].
    report_on_event : bool
        Call ``agent.report()`` on each event in the data stream. Default to True.
    report_on_stop : bool
        Call ``agent.report()`` at the stop document in the data stream. Default to True.
    stream_name : str
        Stream name contained in the descriptor document. Default to "primary".

    Examples
    --------
    >>> class Agent:
    >>>     self.positions = []
    >>>     self.intensities = []
    >>>     def tell(self, position, intensity):
    >>>         self.positions.append(position)
    >>>         self.intensities.append(intensity)
    >>>     def report(self):
    >>>         if self.intensities[-1] < 0: print(f"WARNING, negative intensity at {self.positions[-1]}.")
    >>> agent = Agent()
    >>> RE.subscribe(AgentCallback(agent, independent_key="position", dependent_key="intensity"))
    """

    def __init__(
        self,
        *agents,
        independent_key: str,
        dependent_key: str,
        report_on_event: bool = True,
        report_on_stop: bool = True,
        stream_name: str = "primary",
    ):
        super().__init__()
        self.agents = list()
        for agent in agents:
            if self.agent_validation(agent):
                self.agents.append(agent)
            else:
                warn(f"Agent did not pass validation and will not be included in the callback:\n{agent}")

        self.independent_key = independent_key
        self.dependent_key = dependent_key
        self.report_on_event = report_on_event
        self.report_on_stop = report_on_stop
        self.stream = stream_name
        self._descriptors = {}

    @staticmethod
    def agent_validation(agent):
        passing = True
        tell = getattr(agent, "tell", None)
        if not callable(tell):
            passing = False
            warn("Agent missing callable tell method.")
        report = getattr(agent, "report", None)
        if not callable(report):
            passing = False
            warn("Agent missing callable report method.")
        return passing

    def descriptor(self, doc):
        """Track descriptor documents to only send correct stream to agents"""
        self._descriptors[doc["uid"]] = doc

    def event(self, doc):
        descriptor = self._descriptors[doc["descriptor"]]
        if descriptor.get("name") != self.stream:
            return

        x = doc["data"][self.independent_key]
        y = doc["data"][self.dependent_key]
        for agent in self.agents:
            agent.tell(x, y)

        if self.report_on_event:
            for agent in self.agents:
                agent.report()

    def stop(self, doc):
        if self.report_on_stop:
            for agent in self.agents:
                agent.report()
