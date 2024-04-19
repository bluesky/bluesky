from enum import Enum

import pytest

from bluesky._vendor.super_state_machine import utils


class StatesEnum(Enum):
    ONE = "one"
    TWO = "two"
    THREE = "three"
    FOUR = "four"


class OtherEnum(Enum):
    ONE = "one"


class NonUniqueEnum(Enum):
    ONE = "one"
    TWO = "two"
    THREE = "one"


class CollidingEnum(Enum):
    OPEN = "open"
    OPENING = "opening"
    CLOSE = "close"
    CLOSED = "closed"


def test_translator():
    trans = utils.EnumValueTranslator(StatesEnum)
    assert trans.translate("one") == StatesEnum.ONE
    assert trans.translate("two") == StatesEnum.TWO
    assert trans.translate("three") == StatesEnum.THREE
    assert trans.translate("four") == StatesEnum.FOUR


def test_translator_for_wrong_values():
    trans = utils.EnumValueTranslator(StatesEnum)
    with pytest.raises(ValueError):
        trans.translate("a")
    with pytest.raises(ValueError):
        trans.translate("x")
    with pytest.raises(ValueError):
        trans.translate("threex")
    with pytest.raises(ValueError):
        trans.translate("threx")
    with pytest.raises(ValueError):
        trans.translate("fake")


def test_translator_for_enum_value():
    trans = utils.EnumValueTranslator(StatesEnum)
    assert trans.translate(StatesEnum.ONE) is StatesEnum.ONE
    assert trans.translate(StatesEnum.TWO) is StatesEnum.TWO
    with pytest.raises(ValueError):
        trans.translate(OtherEnum.ONE)


def test_translator_doesnt_accept_non_unique_enums():
    with pytest.raises(ValueError):
        utils.EnumValueTranslator(NonUniqueEnum)
