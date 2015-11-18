from nose.tools import assert_equal, assert_is, assert_raises
from bluesky.datanaming import ConditionalFormat


class TestConditionalFormat:

    def test_validate(self):
        validate = ConditionalFormat.validate
        # pass for paired markers or any markers in format fields
        assert_is(None, validate("{first}< {last}>"))
        assert_is(None, validate("{first} {last}"))
        assert_is(None, validate("{first<5}"))
        assert_is(None, validate("{first>99}"))
        # fail for unbalanced or nested markers
        assert_raises(ValueError, validate, "<{first}")
        assert_raises(ValueError, validate, "{first}>")
        assert_raises(ValueError, validate, "{first} <<{last}>>")
        assert_raises(ValueError, validate, "<{first}> >{last}<")
        return


    def test___str__(self):
        cfmt = ConditionalFormat("<foo>bar")
        assert_equal("<foo>bar", str(cfmt))
        assert_equal("<foo>bar", cfmt.s)
        return


    def test_format(self):
        cfmt = ConditionalFormat("<foo>bar")
        assert_equal('foobar', cfmt.format())
        cfmt = ConditionalFormat("{first}< {mi}.> {last}")
        assert_raises(KeyError, cfmt.format)
        assert_equal('John Doe', cfmt.format(first='John', last='Doe'))
        assert_equal('John M. Doe', cfmt.format(
            first='John', last='Doe', mi='M'))
        assert_raises(KeyError, cfmt.format, first='John', mi='M')
        class _Entry:
            pass
        e = _Entry()
        e.first = "John"
        # raise exception for missing attributes in standard segments
        cfmt = ConditionalFormat("{e.first}< {e.mi}.> {e.last}")
        assert_raises(AttributeError, cfmt.format, e=e)
        # omit when attribute is missing in a conditional segment
        e.last = "Doe"
        assert_equal('John Doe', cfmt.format(e=e))
        e.mi = "M"
        assert_equal('John M. Doe', cfmt.format(e=e))
        return

# class TestConditionalFormat
