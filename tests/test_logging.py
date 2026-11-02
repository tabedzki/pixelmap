"""GUI logging, traceback delivery and production notifier contracts."""

import logging
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pixelmap import error_reporting
from pixelmap.gui import watchdog


@pytest.fixture
def isolated_logging():
    names = ('pixelmap', 'bokeh', 'tornado')
    saved = [(logging.getLogger(n), logging.getLogger(n).handlers[:],
              logging.getLogger(n).level, logging.getLogger(n).propagate) for n in names]
    logger = logging.getLogger('pixelmap')
    logger.handlers = []
    yield logger
    for logger, handlers, level, propagate in saved:
        for handler in logger.handlers:
            if handler not in handlers:
                handler.close()
        logger.handlers = handlers
        logger.setLevel(level)
        logger.propagate = propagate


def configure():
    from pixelmap.gui.logging_config import configure_logging
    configure_logging()


def test_notifier_contract(caplog):
    # Production's docker-logs notifier matches these exact substrings.
    caplog.set_level(logging.WARNING)
    now = [0.0]
    stall = watchdog.StallWatchdog(clock=lambda: now[0])
    stall.beat()
    now[0] = 6.0
    stall.check()
    stall.beat()
    watchdog.MemoryWatch(100).check(101)
    for text in ('[watchdog] event loop blocked', '[watchdog] event loop recovered',
                 '[watchdog] RSS'):
        assert any(text in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize('raw,level', [(None, 20), ('debug', 10), ('WARNING', 30),
                                     ('30', 30), ('', 20), ('bogus', 20), ('-1', 20)])
def test_configuration(isolated_logging, monkeypatch, capsys, raw, level):
    if raw is None:
        monkeypatch.delenv('PIXELMAP_LOG_LEVEL', raising=False)
    else:
        monkeypatch.setenv('PIXELMAP_LOG_LEVEL', raw)
    root = logging.getLogger()
    original = (root.level, root.handlers[:])
    configure()
    handlers = isolated_logging.handlers[:]
    configure()
    assert isolated_logging.handlers == handlers
    assert len(handlers) == 2
    assert isolated_logging.level == level
    assert isolated_logging.propagate is False
    assert (root.level, root.handlers) == original
    assert logging.getLogger('bokeh').level == logging.ERROR
    assert logging.getLogger('tornado').level == logging.ERROR
    stderr = capsys.readouterr().err
    assert stderr.count('Invalid PIXELMAP_LOG_LEVEL') == (1 if raw in ('', 'bogus', '-1') else 0)


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


def test_watchdog_default_visible(isolated_logging, monkeypatch, caplog, capsys):
    monkeypatch.delenv('PIXELMAP_LOG_LEVEL', raising=False)
    configure()
    isolated_logging.addHandler(caplog.handler)
    watchdog.MemoryWatch(100).check(101)
    assert '[watchdog] RSS' in caplog.text
    assert capsys.readouterr().err.count('[watchdog] RSS') == 1


def test_forwarding(isolated_logging, monkeypatch):
    monkeypatch.setenv('PIXELMAP_ERROR_WEBHOOK_URL', 'https://example.test/errors')
    post = MagicMock()
    monkeypatch.setattr(error_reporting.urllib.request, 'urlopen', post)
    reporter = error_reporting.ErrorReporter(synchronous=True)
    monkeypatch.setattr(error_reporting, '_REPORTER', reporter)
    toast = MagicMock()
    monkeypatch.setattr(error_reporting, '_toast', toast)
    configure()
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


def test_malformed_imro_upload_is_not_forwarded(isolated_logging, monkeypatch, caplog):
    # A bad user upload is the user's problem, not a server fault: keep the
    # traceback in the logs, toast the user, but don't page the webhook.
    from pixelmap.gui import gui
    monkeypatch.setenv('PIXELMAP_ERROR_WEBHOOK_URL', 'https://example.test/errors')
    post = MagicMock()
    monkeypatch.setattr(error_reporting.urllib.request, 'urlopen', post)
    monkeypatch.setattr(error_reporting, '_REPORTER', error_reporting.ErrorReporter(synchronous=True))
    notify = MagicMock()
    monkeypatch.setattr(gui, '_notify', notify)
    monkeypatch.setattr(gui.imro, 'parse_imro_file', lambda text: [('NP1000', 384)])
    monkeypatch.setattr(gui.imro, 'parse_imro_list', MagicMock(side_effect=ValueError('bad imro')))
    configure()
    caplog.set_level(logging.DEBUG, logger='pixelmap')
    logging.getLogger('pixelmap').addHandler(caplog.handler)
    fake = SimpleNamespace(imro_file_loader=SimpleNamespace(value=b'(0,384)', filename='x.imro'))
    gui.ChannelmapGUI.apply_uploaded_imro(fake)
    record = next(r for r in caplog.records if 'uploaded imro' in r.getMessage())
    assert record.levelno == logging.WARNING
    assert record.exc_info[0] is ValueError
    notify.assert_called_once()
    post.assert_not_called()
