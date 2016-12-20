from PyQt5.QtWidgets import QTreeWidgetItem, QMainWindow, QTreeWidget
from PyQt5 import QtWidgets
from bluesky import Msg
import pprint
from bluesky.plan_tools import to_nested


def make_row(msg):
    kwarg_str = '{!r}'.format(msg.kwargs)
    if len(kwarg_str) > 50:
        kwarg_str = '{...}'
    child = QTreeWidgetItem([msg.command,
                             msg.obj.name if msg.obj is not None else '',
                             '{}'.format(msg.args),
                             kwarg_str])
    child.setToolTip(3, pprint.pformat(msg.kwargs))
    return child


def fill_item(item, value):
    item.setExpanded(True)
    if isinstance(value, dict):
        for key, val in sorted(value.items()):
            child = make_row(val[0])
            item.addChild(child)
            fill_item(child, val[1:])
    elif isinstance(value, list):
        for val in value:
            fill_item(item, val)
    else:
        child = make_row(value)

        item.addChild(child)


def fill_widget(widget, value):
    widget.clear()
    fill_item(widget.invisibleRootItem(), value)


def plan_writer():
    window = QMainWindow()
    mw = QtWidgets.QWidget()
    ed_pannel = QtWidgets.QWidget()
    ed_layout = QtWidgets.QVBoxLayout()
    ed_pannel.setLayout(ed_layout)

    w = QTreeWidget()
    w.setColumnCount(4)
    w.setHeaderLabels(Msg._fields)
    w.setAlternatingRowColors(True)

    ed = QtWidgets.QPlainTextEdit()
    ed_layout.addWidget(ed)
    # go_button = QtWidgets.QPushButton('display `target`')
    # ed_layout.addWidget(go_button)

    layout = QtWidgets.QHBoxLayout()
    layout.addWidget(w)
    layout.addWidget(ed_pannel)
    mw.setLayout(layout)

    window.setCentralWidget(mw)

    def update_tree():
        lcls = {}
        try:
            exec(ed.toPlainText(), globals(), lcls)
            ret = to_nested(lcls['target'])
        except Exception as e:
            pass
        else:
            fill_widget(w, ret)
            w.resizeColumnToContents(0)
            w.resizeColumnToContents(1)

    ed.textChanged.connect(update_tree)
    # go_button.pressed.connect(update_tree)
    window.show()
    return window
