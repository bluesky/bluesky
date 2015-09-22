"""
Check if expected databases and hardware are alive.
"""
import os
import logging
from collections import namedtuple


logger = logging.getLogger(__name__)


def connect_mds_mongodb():
    import metadatastore.conf
    _connect_mongodb('metadatastore', metadatastore.conf.connection_config)


def connect_fs_mongodb():
    import filestore.conf
    _connect_mongodb('filestore', filestore.conf.connection_config)


def _connect_mongodb(name, connection_config):
    import pymongo
    cc = connection_config
    logger.debug("Attempting to connect to %s mongodb at host %s "
                 "and port %d", name, cc['host'], cc['port'])
    try:
        mc = pymongo.MongoClient(cc['host'], cc['port'])
    except Exception:
        logger.debug("Failed to connect to %s mongodb.", name)
        raise
    else:
        logger.debug("Successfully connected to %s mongodb.", name)
    logger.debug("Attemping to find %s database called %s in mongodb.", name,
                 cc['database'])
    if cc['database'] in mc.database_names():
        logger.debug("Database for %s successfully found.", name)
    else:
        logger.debug("Database for %s not found.", name)
        raise RuntimeError("Database for %s called %s not found." %
                           (name, cc['database']))


def connect_olog():
    import pyOlog
    import pyOlog.conf
    paths = pyOlog.conf.Config.conf_files
    found_file = False
    for path in paths:
        logger.debug("Attemping to find olog config file at path %s", path)
        if os.path.isfile(path):
            logger.debug("Olog config file successfully found.")
            found_file = True
            break
        else:
            logger.debug("Olog config file not found.")
    if not found_file:
        logger.debug("Olog config file was not found at any of "
                     "these paths: %r", paths)
        raise RuntimeError("Olog config file was not found at any of "
                           "these paths: %r" % paths)
    logger.debug("Attempting to create OlogClient.")
    try:
        pyOlog.OlogClient()
    except Exception:
        logger.debug("Failed to create OlogClient.")
        raise RuntimeError("Failed to create OlogClient.")
    else:
        logger.debug("Successfully created OlogClient.")


def connect_channelarchiver(url):
    """Try to access the channel archiver through HTTP.

    Parameters
    ----------
    url : string
        For example, 'http://xf23id-ca.cs.nsls2.local:4800'
    """
    import requests
    logger.debug("Attempting to connect to the channel archiver at: %s", url)
    response = requests.get(url)
    if not response:
        logger.debug("Failed to connect to the channel archiver.")
        raise RuntimeError("Failed to connect to the channel archiver at "
                           "%s. Response: %r" % (url, response))
    logger.debug("Successfully connected to the channel archiver.")


usage_tuple = namedtuple('usage', 'total used free')


def check_storage(path, required_free):
    # Adapated from http://stackoverflow.com/a/7285483
    st = os.statvfs(path)
    free = st.f_bavail * st.f_frsize
    total = st.f_blocks * st.f_frsize
    used = (st.f_blocks - st.f_bfree) * st.f_frsize
    usage = usage_tuple(total, used, free)
    logger.debug('Disk usage at %s is %s', path, usage)
    if free < required_free:
        raise RuntimeError("Disk at %s has %d bytes free" % (path, free))


def _skeptical_caget(pv):
    # Note: pv must be a scalar PV
    import epics
    value = epics.caget(pv)
    if value is None:
        raise RuntimeError("Failed to connect to pv %s " % pv)
    return value


def connect_pv(pv):
    _skeptical_caget(pv)

def assert_pv_equal(pv, value):
    actual = _skeptical_caget(pv)
    if actual != value:
        raise AssertionError("PV %s returned %f but %f was expected." %
                             (pv, actual, value))


def assert_pv_greater(pv, value):
    actual = _skeptical_caget(pv)
    if not actual > value:
        raise AssertionError("PV %s returned %f but a value greater than %f "
                             " was expected." % (pv, actual, value))


def assert_pv_less(pv, value):
    actual = _skeptical_caget(pv)
    if not actual < value:
        raise AssertionError("PV %s returned %f but a value less than %f "
                             " was expected." % (pv, actual, value))


def assert_pv_in_band(pv, low, high):
    actual = _skeptical_caget(pv)
    if not low < actual < high:
        raise AssertionError("PV %s returned %f but a value between %f and %f "
                             " was expected." % (pv, actual, low, high))


def assert_pv_out_of_band(pv, low, high):
    actual = _skeptical_caget(pv)
    if not ((actual < low) or (actual > high)):
        raise AssertionError("PV %s returned %f but a value outsdie of the "
                             "range from %f to %f  was expected." %
                             (pv, actual, low, high))
