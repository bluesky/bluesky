def log(name, doc):
    # This only applies to 'start' Documents.
    if doc != 'start':
        return
    input_message, = msg.args
    output = """
Header uid: {{uid!r}}

Scan Plan
---------
{input_message}

Metadata
--------
{{metadata!r}}
""".format(input_message=t

    d = doc
    self.logbook(log_message, d)


def _run_engine_log_template(metadata):
    template = []
    for key in metadata:
        template.append("{key}: {{{key}}}".format(key=key))
    return '\n'.join(template)


def _logmsg(md):

    call_str = _call_str(md['plan_type'], md['plan_args'])

    msgs = ['Scan Class: {scn_cls}', '']
    for k, v in md['plan_args'].items():
        msgs.append('{k}: {{{k}!r}}'.format(k=k))
    msgs.append('')
    msgs.append('To call:')
    msgs.extend(call_str)
    return msgs


def logmsg(self):
    msgs = self._logmsg()
    return '\n'.join(msgs)


def logdict(self):
    out_dict = {k: getattr(self, k) for k in self._fields}
    out_dict['scn_cls'] = self.__class__.__name__
    return out_dict


def _call_str(plan_type, plan_args):
    args = []
    for k, v in plan_args.items():
        args.append("{k}={{{k}!r}}".format(k=k))

    return ["RE({{scn_cls}}({args}))".format(args=', '.join(args)), ]
