from io import StringIO
from pprint import pformat
TEMPLATES = {}
TEMPLATES['long'] = """
{{- start.plan_type }} ['{{ start.uid[:6] }}'] (scan num: {{ start.scan_id }})

Scan Plan
---------
{{ start.plan_type }}
{%- for k, v in start.plan_args | dictsort %}
    {{ k }}: {{ v }}
{%-  endfor %}

{% if 'signature' in start -%}
Call:
    {{ start.signature }}
{% endif %}
Metadata
--------
{% for k, v in start.items() -%}
{%- if k not in ['plan_type', 'plan_args'] -%}{{ k }} : {{ v }}
{% endif -%}
{%- endfor -%}"""
TEMPLATES['desc'] = """
{{- start.plan_type }} ['{{ start.uid[:6] }}'] (scan num: {{ start.scan_id }})"""
TEMPLATES['call'] = """RE({{ start.plan_type }}(
{%- for k, v in start.plan_args.items() %}{%- if not loop.first %}   {% endif %}{{ k }}={{ v }}
{%- if not loop.last %},
{% endif %}{% endfor %}))
"""

def logbook_cb_factory(logbook_func, desc_template=None, long_template=None):
    """Create a logbook run_start callback

    The returned function is suitable for registering as
    a 'start' callback on the the BlueSky run engine.

    Parameters
    ----------

    logbook_func : callable
        The required signature is ::

            def logbok_func(text=None, logbooks=None, tags=None, properties=None,
                            attachments=None, verify=True, ensure=False):
                '''

                Parameters
                ----------
                text : string
                    The body of the log entry.
                logbooks : string or list of strings
                    The logbooks which to add the log entry to.
                tags : string or list of strings
                    The tags to add to the log entry.
                properties : dict of property dicts
                    The properties to add to the log entry
                attachments : list of file like objects
                    The attachments to add to the log entry
                verify : bool
                    Check that properties, tags and logbooks are in the Olog
                    instance.
                ensure : bool
                    If a property, tag or logbook is not in the Olog then
                    create the property, tag or logbook before making the log
                    entry. Seting ensure to True will set verify to False.

                '''
                pass

        This matches the API on `SimpleOlogClient.log`

    """
    import jinja2
    env = jinja2.Environment()
    if long_template is None:
        long_template = TEMPLATES['long']
    if desc_template is None:
        desc_template = TEMPLATES['desc']
    # It seems that the olog only has one text field, which it calls
    # `text` on the python side and 'description' on the olog side.
    # There are some CSS applications that try to shove the entire
    # thing into a single line.  We work around this by doing two
    # strings, a long one which will get put in a as an attachment
    # and a short one to go in as the 'text' which will be used as the
    # description
    long_msg = env.from_string(long_template)
    desc_msg = env.from_string(desc_template)

    def lbcb(name, doc):
        # This only applies to 'start' Documents.
        if name != 'start':
            return

        atch = StringIO(long_msg.render(start=doc))
        desc = desc_msg.render(start=doc)
        logbook_func(text=desc, properties={'start':doc},
                attachments=[atch],
                ensure=True)
    return lbcb

def call_str(start, call_template=None):
    """Given a start document generate an evalable call scring

    The default template assumes that `plan_args` and `plan_type`
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

    This is not necessarily the best example of how to customize the log book,
    but it is something that fits the needs of CHX

    Example
    -------
    # add this callback to the run engine
    >>> gs.RE.subscribe('start', OlogCallback())
    # turn off the default logger
    >>> gs.RE.logbook = None
    """
    def __init__(self):
        # import a whole mess of stuff when this thing gets created
        from pyOlog import SimpleOlogClient
        self.client = SimpleOlogClient()

    def start(self, doc):
        from IPython import get_ipython
        commands = list(get_ipython().history_manager.get_range())
        document_content = ('%s: %s\n\n'
                            'RunStart Document\n'
                            '-----------------\n'
                            '%s' % (doc['scan_id'],
                                    commands[-1][2],
                                    pformat(doc)))
        olog_status = self.client.log(document_content, logbooks='Data Acquisition')
        logger.debug('client.log returned %s' % olog_status)
