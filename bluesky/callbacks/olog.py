from io import StringIO
from pprint import pformat
import logging
from . import CallbackBase
from collections import defaultdict

logger = logging.getLogger(__name__)

TEMPLATES = {}
TEMPLATES['long'] = """
{{- start.plan_name }} ['{{ start.uid[:6] }}'] (scan num: {{ start.scan_id }})

Scan Plan
---------
{{ start.plan_name }}
{% if 'plan_args' in start %}
    {%- for k, v in start.plan_args | dictsort %}
        {{ k }}: {{ v }}
    {%-  endfor %}
{% endif %}

{% if 'signature' in start -%}
Call:
    {{ start.signature }}
{% endif %}
Metadata
--------
{% for k, v in start.items() -%}
{%- if k not in ['plan_name', 'plan_args'] -%}{{ k }} : {{ v }}
{% endif -%}
{%- endfor -%}"""

TEMPLATES['desc'] = """
{{- start.plan_name }} ['{{ start.uid[:6] }}'] (scan num: {{ start.scan_id }})"""

TEMPLATES['call'] = """RE({{ start.plan_name }}(
{%- for k, v in start.plan_args.items() %}{%- if not loop.first %}   {% endif %}{{ k }}={{ v }}
{%- if not loop.last %},
{% endif %}{% endfor %}))
"""


def logbook_cb_factory(logbook_func, desc_template=None, long_template=None,
                       desc_dispatch=None, long_dispatch=None):
    """Create a logbook run_start callback

    The returned function is suitable for registering as
    a 'start' callback on the the BlueSky run engine.

    Parameters
    ----------
    logbook_func : callable
        The required signature should match the API ``SimpleOlogClient.log``.
        It is:

        .. code-block:: python

            logbook_func(text=None, logbooks=None, tags=None,
                         properties=None, attachments=None, verify=True,
                         ensure=False)

    desc_template : str, optional
        A jinja2 template to be used for the description line in olog.  This is
        the default used if the plan_name does not map to a more specific one.

    long_template : str, optional
        A jinja2 template to be used for the attachment in olog.  This is
        the default used if the plan_name does not map to a more specific one.

    desc_dispatch, long_dispatch : mapping, optional
        Mappings between 'plan_name' to jinja2 templates to use for the
        description and attachments respectively.
    """
    import jinja2
    env = jinja2.Environment()
    if long_template is None:
        long_template = TEMPLATES['long']
    if desc_template is None:
        desc_template = TEMPLATES['desc']

    if desc_dispatch is None:
        desc_dispatch = {}
    if long_dispatch is None:
        long_dispatch = {}

    # It seems that the olog only has one text field, which it calls
    # `text` on the python side and 'description' on the olog side.
    # There are some CSS applications that try to shove the entire
    # thing into a single line.  We work around this by doing two
    # strings, a long one which will get put in a as an attachment
    # and a short one to go in as the 'text' which will be used as the
    # description
    long_msg = env.from_string(long_template)
    desc_msg = env.from_string(desc_template)

    desc_dispatch = defaultdict(lambda: desc_msg,
                                {k: env.from_string(v)
                                 for k, v in desc_dispatch.items()})
    long_dispatch = defaultdict(lambda: long_msg,
                                {k: env.from_string(v)
                                 for k, v in long_dispatch.items()})

    def lbcb(name, doc):
        # This only applies to 'start' Documents.
        if name != 'start':
            return
        plan_name = doc.get('plan_name', '')
        body = long_dispatch[plan_name]
        desc = desc_dispatch[plan_name]
        atch = StringIO(body.render(start=doc))
        # monkey-patch a 'name' attribute onto StringIO
        atch.name = 'long_description.txt'
        desc = desc.render(start=doc)
        logbook_func(text=desc, attachments=[atch], ensure=True)
    return lbcb


def call_str(start, call_template=None):
    """Given a start document generate an evalable call scring

    The default template assumes that `plan_args` and `plan_name`
    are at the top level of the document.

    Parameter
    ---------
    start : dict
        A document which follows the runstart schema

    call_template : str, optional
        A jinja2 template rendered with `cr.render(start=start)`

        If not provided defaults to `CALL_TEMPLATE`
    """
    import jinja2
    env = jinja2.Environment()
    if call_template is None:
        call_template = TEMPLATES['call']
    call_renderer = env.from_string(call_template)
    return call_renderer.render(start=start)


class OlogCallback(CallbackBase):
    """Example callback to customize the logbook.

    This callback publishes the most recent IPython command (which of course
    is not guaranteed to be the one that initiated the run in question) and
    the full RunStart Document.

    Example
    -------
    # add this callback to the run engine
    >>> gs.RE.subscribe(OlogCallback(), 'start')
    # turn off the default logger
    >>> gs.RE.logbook = None
    """
    def __init__(self, logbook):
        self.logbook = logbook
        from pyOlog import SimpleOlogClient
        self.client = SimpleOlogClient()
        # Check at init time we are in an IPython session.
        from IPython import get_ipython

    def start(self, doc):
        from IPython import get_ipython
        commands = list(get_ipython().history_manager.get_range())
        document_content = ('%s: %s\n\n'
                            'RunStart Document\n'
                            '-----------------\n'
                            '%s' % (doc['scan_id'],
                                    commands[-1][2],
                                    pformat(doc)))
        olog_status = self.client.log(document_content, logbooks=self.logbook)
        logger.debug('client.log returned %s' % olog_status)
        super().start(doc)
