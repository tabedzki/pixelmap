"""GUI logging, traceback delivery and production notifier contracts."""

import io
import logging
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pixelmap import error_reporting
from pixelmap.gui import logging_config, watchdog
from pixelmap.gui.logging_config import configure_logging


@pytest.fixture
def isolated_logging():
    loggers = {n: logging.getLogger(n) for n in ('pixelmap', 'bokeh', 'tornado')}
    saved = {n: (lg.handlers[:], lg.level, lg.propagate) for n, lg in loggers.items()}
    logger = loggers['pixelmap']
    logger.handlers = []
    yield logger
    for name, (handlers, level, propagate) in saved.items():
        loggers[name].handlers = handlers
        loggers[name].setLevel(level)
        loggers[name].propagate = propagate


@pytest.fixture
def stderr(monkeypatch):
    # The module-level handler binds sys.stderr at import; capture it separately.
    buffer = io.StringIO()
    monkeypatch.setattr(logging_config._STDERR, 'stream', buffer)
    return buffer


@pytest.fixture
def webhook(monkeypatch):
    monkeypatch.setenv('PIXELMAP_ERROR_WEBHOOK_URL', 'https://example.test/errors')
    post = MagicMock()
    monkeypatch.setattr(error_reporting.urllib.request, 'urlopen', post)
    monkeypatch.setattr(error_reporting, '_REPORTER', error_reporting.ErrorReporter(synchronous=True))
    return post


@pytest.mark.parametrize('raw,level', [(None, 20), ('debug', 10), ('WARNING', 30),
                                     (' error ', 40), ('30', 20), ('', 20), ('bogus', 20),
                                     ('-1', 20)])
def test_level_parsing(isolated_logging, monkeypatch, raw, level):
    if raw is None:
        monkeypatch.delenv('PIXELMAP_LOG_LEVEL', raising=False)
    else:
        monkeypatch.setenv('PIXELMAP_LOG_LEVEL', raw)
    configure_logging()
    assert isolated_logging.level == level


def test_configuration_invariants(isolated_logging, monkeypatch, stderr):
    monkeypatch.setenv('PIXELMAP_LOG_LEVEL', 'bogus')
    root = logging.getLogger()
    original = (root.level, root.handlers[:])
    configure_logging()
    handlers = isolated_logging.handlers[:]
    configure_logging()
    assert isolated_logging.handlers == handlers
    assert len(handlers) == 2
    assert isolated_logging.propagate is False
    assert (root.level, root.handlers) == original
    assert logging.getLogger('bokeh').level == logging.ERROR
    assert logging.getLogger('tornado').level == logging.ERROR
    # The invalid-level warning is emitted on first install only.
    assert stderr.getvalue().count('Invalid PIXELMAP_LOG_LEVEL') == 1


def test_library_import_configures_nothing():
    result = subprocess.run([sys.executable, '-c', '''
import logging
root = logging.getLogger()
before = root.handlers[:]
import pixelmap
assert root.handlers == before
assert not logging.getLogger('pixelmap').handlers
'''], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_watchdog_default_visible(isolated_logging, monkeypatch, caplog, stderr):
    monkeypatch.delenv('PIXELMAP_LOG_LEVEL', raising=False)
    configure_logging()
    isolated_logging.addHandler(caplog.handler)
    watchdog.MemoryWatch(100).check(101)
    assert '[watchdog] RSS' in caplog.text
    assert stderr.getvalue().count('[watchdog] RSS') == 1


def test_forwarding(isolated_logging, monkeypatch, webhook):
    post = webhook
    toast = MagicMock()
    monkeypatch.setattr(error_reporting, '_toast', toast)
    configure_logging()
    log = logging.getLogger('pixelmap.gui.test')
    try:
        raise ValueError('forward me')
    except ValueError as exc:
        log.exception('callback failed')
        log.exception('same failure')
        assert post.call_count == 1
        toast.assert_not_called()
        error_reporting.report_error(exc)
        assert post.call_count == 1
    # Even with dedupe disabled, report_error's own log must not forward again.
    monkeypatch.setenv('PIXELMAP_ERROR_WEBHOOK_DEDUPE_SECONDS', '0')
    error_reporting.report_error(RuntimeError('direct'))
    assert post.call_count == 2
    log.error('plain error without exception')
    assert post.call_count == 2


def test_cleanup_traceback(monkeypatch, caplog):
    from pixelmap.gui import gui
    doc = object()
    view = SimpleNamespace(_cleanup=MagicMock(side_effect=RuntimeError('cleanup broken')))
    monkeypatch.setattr(gui.pn.state, '_views', {'test': (view, None, doc, None)})
    gui._release_dead_session_views(SimpleNamespace(_document=doc))
    assert not gui.pn.state._views
    record = next(r for r in caplog.records if 'View cleanup failed' in r.getMessage())
    assert record.exc_info[0] is RuntimeError


def test_registry_fallback_traceback(monkeypatch, caplog):
    from pixelmap.anatomy import atlas
    monkeypatch.setattr(atlas, 'get_downloaded_atlases', lambda: ['test'])
    # A failed metadata lookup preserves the not-downloaded fallback.
    monkeypatch.setattr(atlas, 'get_atlas', MagicMock(side_effect=OSError('bad metadata')))
    caplog.set_level(logging.DEBUG, logger=atlas.__name__)
    assert atlas.is_downloaded('test') is False
    assert any(r.exc_info and r.exc_info[0] is OSError for r in caplog.records)


def test_malformed_imro_upload_is_not_forwarded(isolated_logging, monkeypatch, caplog, webhook):
    # A bad user upload is the user's problem, not a server fault: keep the
    # traceback in the logs, toast the user, but don't page the webhook.
    from pixelmap.gui import gui
    post = webhook
    notify = MagicMock()
    monkeypatch.setattr(gui, '_notify', notify)
    monkeypatch.setattr(gui.imro, 'parse_imro_file', lambda text: [('NP1000', 384)])
    monkeypatch.setattr(gui.imro, 'parse_imro_list', MagicMock(side_effect=ValueError('bad imro')))
    configure_logging()
    caplog.set_level(logging.DEBUG, logger='pixelmap')
    logging.getLogger('pixelmap').addHandler(caplog.handler)
    fake = SimpleNamespace(imro_file_loader=SimpleNamespace(value=b'(0,384)', filename='x.imro'))
    gui.ChannelmapGUI.apply_uploaded_imro(fake)
    record = next(r for r in caplog.records if 'uploaded imro' in r.getMessage())
    assert record.levelno == logging.WARNING
    assert record.exc_info[0] is ValueError
    notify.assert_called_once()
    post.assert_not_called()
