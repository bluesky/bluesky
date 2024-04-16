===================
Super State Machine
===================

.. image:: https://badge.fury.io/py/super_state_machine.png
    :target: http://badge.fury.io/py/super_state_machine

.. image:: https://travis-ci.org/beregond/super_state_machine.png?branch=master
        :target: https://travis-ci.org/beregond/super_state_machine

.. image:: https://pypip.in/d/super_state_machine/badge.png
        :target: https://pypi.python.org/pypi/super_state_machine


Super State Machine gives you utilities to build finite state machines.

* Free software: BSD license
* Documentation: https://super_state_machine.readthedocs.org
* Source: https://github.com/beregond/super_state_machine

Features
--------

* Fully tested with Python 2.7, 3.3, 3.4 and PyPy.

* Create finite state machines:

  .. code-block:: python

    >>> from enum import Enum

    >>> from super_state_machine import machines


    >>> class Task(machines.StateMachine):
    ...
    ...    state = 'draft'
    ...
    ...    class States(Enum):
    ...
    ...         DRAFT = 'draft'
    ...         SCHEDULED = 'scheduled'
    ...         PROCESSING = 'processing'
    ...         SENT = 'sent'
    ...         FAILED = 'failed'

    >>> task = Task()
    >>> task.is_draft
    False
    >>> task.set_draft()
    >>> task.state
    'draft'
    >>> task.state = 'scheduled'
    >>> task.is_scheduled
    True
    >>> task.state = 'process'
    >>> task.state
    'processing'
    >>> task.state = 'wrong'
    *** ValueError: Unrecognized value ('wrong').

* Define allowed transitions graph, define additional named transitions
  and checkers:

  .. code-block:: python

    >>> class Task(machines.StateMachine):
    ...
    ...     class States(Enum):
    ...
    ...         DRAFT = 'draft'
    ...         SCHEDULED = 'scheduled'
    ...         PROCESSING = 'processing'
    ...         SENT = 'sent'
    ...         FAILED = 'failed'
    ...
    ...     class Meta:
    ...
    ...         allow_empty = False
    ...         initial_state = 'draft'
    ...         transitions = {
    ...             'draft': ['scheduled', 'failed'],
    ...             'scheduled': ['failed'],
    ...             'processing': ['sent', 'failed']
    ...         }
    ...         named_transitions = [
    ...             ('process', 'processing', ['scheduled']),
    ...             ('fail', 'failed')
    ...         ]
    ...         named_checkers = [
    ...             ('can_be_processed', 'processing'),
    ...         ]

    >>> task = Task()
    >>> task.state
    'draft'
    >>> task.process()
    *** TransitionError: Cannot transit from 'draft' to 'processing'.
    >>> task.set_scheduled()
    >>> task.can_be_processed
    True
    >>> task.process()
    >>> task.state
    'processing'
    >>> task.fail()
    >>> task.state
    'failed'

  Note, that third argument restricts from which states transition will be
  **added** to allowed (in case of ``process``, new allowed transition will be
  added, from 'scheduled' to 'processing'). No argument means all available
  states, ``None`` or empty list won't add anything beyond defined ones.

* Use state machines as properties:

.. code-block:: python

  >>> from enum import Enum

  >>> from super_state_machine import machines, extras


  >>> class Lock(machine.StateMachine):

  ...     class States(Enum):
  ...
  ...         OPEN = 'open'
  ...         LOCKED = 'locked'
  ...
  ...     class Meta:
  ...
  ...         allow_empty = False
  ...         initial_state = 'locked'
  ...         named_transitions = [
  ...             ('open', 'open'),
  ...             ('lock', 'locked'),
  ...         ]


  >>> class Safe(object):
  ...
  ...     lock1 = extras.PropertyMachine(Lock)
  ...     lock2 = extras.PropertyMachine(Lock)
  ...     lock3 = extras.PropertyMachine(Lock)
  ...
  ...     locks = ['lock1', 'lock2', 'lock3']
  ...
  ...     def is_locked(self):
  ...          locks = [getattr(self, lock).is_locked for lock in self.locks]
  ...          return any(locks)
  ...
  ...     def is_open(self):
  ...         locks = [getattr(self, lock).is_open for lock in self.locks]
  ...         return all(locks)

  >>> safe = Safe()
  >>> safe.lock1
  'locked'
  >>> safe.is_open
  False
  >>> safe.lock1.open()
  >>> safe.lock1.is_open
  True
  >>> safe.lock1
  'open'
  >>> safe.is_open
  False
  >>> safe.lock2.open()
  >>> safe.lock3 = 'open'
  >>> safe.is_open
  True
