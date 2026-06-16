# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for easyscience.global_object.logger.

Covers the module-level parsing helpers and the public surface of the
``Logger`` class: the ``level`` property, ``getLogger``, the convenience
delegation methods, the ``at_level`` context manager and the
``suspend``/``resume`` shims.
"""

import logging

import pytest

from easyscience.global_object.logger import _LOG_LEVEL_ENV_VAR
from easyscience.global_object.logger import PACKAGE_LOGGER_NAME
from easyscience.global_object.logger import Logger
from easyscience.global_object.logger import _env_log_level
from easyscience.global_object.logger import _parse_log_level


@pytest.fixture(autouse=True)
def _restore_package_logger_level():
    """Snapshot and restore the package-root logger level around each test.

    ``Logger`` operates on the process-wide ``easyscience`` logger, so a
    test that changes its level would otherwise leak into other tests.
    """
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    previous = package_logger.level
    yield
    package_logger.setLevel(previous)


# ---------------------------------------------------------------------------
# _parse_log_level
# ---------------------------------------------------------------------------


class TestParseLogLevel:
    def test_int_passthrough(self):
        # When Then
        assert _parse_log_level(logging.ERROR) == logging.ERROR
        assert _parse_log_level(42) == 42

    @pytest.mark.parametrize(
        'name, expected',
        [
            ('DEBUG', logging.DEBUG),
            ('INFO', logging.INFO),
            ('WARNING', logging.WARNING),
            ('ERROR', logging.ERROR),
            ('CRITICAL', logging.CRITICAL),
        ],
    )
    def test_level_names(self, name, expected):
        # When Then
        assert _parse_log_level(name) == expected

    def test_name_is_case_insensitive_and_stripped(self):
        # When Then
        assert _parse_log_level('  error ') == logging.ERROR
        assert _parse_log_level('Warning') == logging.WARNING

    def test_numeric_string(self):
        # When Then
        assert _parse_log_level('40') == 40
        assert _parse_log_level('  10 ') == 10

    def test_unknown_name_raises_value_error(self):
        # When Then
        with pytest.raises(ValueError) as exc:
            _parse_log_level('VERBOSE')
        # Expect the message to name the offending value and list valid options
        assert 'VERBOSE' in str(exc.value)
        assert 'WARNING' in str(exc.value)


# ---------------------------------------------------------------------------
# _env_log_level
# ---------------------------------------------------------------------------


class TestEnvLogLevel:
    def test_none_returns_default(self):
        # When Then
        assert _env_log_level(None, default=logging.WARNING) == logging.WARNING

    def test_valid_value_is_parsed(self):
        # When Then
        assert _env_log_level('ERROR', default=logging.WARNING) == logging.ERROR
        assert _env_log_level('10', default=logging.WARNING) == 10

    def test_bad_value_falls_back_to_default(self):
        # When Then — a typo must not raise, it falls back to the default
        assert _env_log_level('NOPE', default=logging.INFO) == logging.INFO


# ---------------------------------------------------------------------------
# Logger.__init__
# ---------------------------------------------------------------------------


class TestLoggerInit:
    def test_default_level_is_warning(self, monkeypatch):
        # Given no env var set
        monkeypatch.delenv(_LOG_LEVEL_ENV_VAR, raising=False)

        # When
        log = Logger()

        # Then
        assert log.level == logging.WARNING
        assert log.logger.name == PACKAGE_LOGGER_NAME

    def test_custom_default_level(self, monkeypatch):
        # Given
        monkeypatch.delenv(_LOG_LEVEL_ENV_VAR, raising=False)

        # When
        log = Logger(log_level=logging.DEBUG)

        # Then
        assert log.level == logging.DEBUG

    def test_env_var_overrides_default(self, monkeypatch):
        # Given the env var requests ERROR
        monkeypatch.setenv(_LOG_LEVEL_ENV_VAR, 'ERROR')

        # When the default would have been WARNING
        log = Logger(log_level=logging.WARNING)

        # Then the env var wins
        assert log.level == logging.ERROR

    def test_invalid_env_var_falls_back_to_default(self, monkeypatch):
        # Given a typo'd env var
        monkeypatch.setenv(_LOG_LEVEL_ENV_VAR, 'NONSENSE')

        # When
        log = Logger(log_level=logging.INFO)

        # Then import does not crash and the default is used
        assert log.level == logging.INFO


# ---------------------------------------------------------------------------
# Logger.level property
# ---------------------------------------------------------------------------


class TestLevelProperty:
    def test_getter_reflects_underlying_logger(self):
        # Given
        log = Logger()
        log.logger.setLevel(logging.CRITICAL)

        # When Then
        assert log.level == logging.CRITICAL

    def test_setter_accepts_int(self):
        # Given
        log = Logger()

        # When
        log.level = logging.DEBUG

        # Then
        assert log.logger.level == logging.DEBUG

    def test_setter_accepts_level_name(self):
        # Given
        log = Logger()

        # When
        log.level = 'ERROR'

        # Then
        assert log.logger.level == logging.ERROR

    def test_setter_raises_on_unknown_name(self):
        # Given
        log = Logger()

        # When Then — the public API must be strict
        with pytest.raises(ValueError):
            log.level = 'LOUD'


# ---------------------------------------------------------------------------
# Logger.getLogger
# ---------------------------------------------------------------------------


class TestGetLogger:
    def test_prefixes_with_package_name(self):
        # Given
        log = Logger()

        # When
        child = log.getLogger('fitting')

        # Then
        assert child.name == f'{PACKAGE_LOGGER_NAME}.fitting'

    def test_does_not_double_prefix(self):
        # Given a name that already starts with the package name
        log = Logger()

        # When
        child = log.getLogger(f'{PACKAGE_LOGGER_NAME}.fitting.bumps')

        # Then
        assert child.name == f'{PACKAGE_LOGGER_NAME}.fitting.bumps'

    def test_child_is_left_at_notset_to_inherit(self):
        # Given
        log = Logger()

        # When
        child = log.getLogger('some_unique_subsystem')

        # Then it inherits configuration from the package root
        assert child.level == logging.NOTSET


# ---------------------------------------------------------------------------
# Convenience delegation methods
# ---------------------------------------------------------------------------


class TestConvenienceMethods:
    @pytest.mark.parametrize(
        'method, level',
        [
            ('debug', logging.DEBUG),
            ('info', logging.INFO),
            ('warning', logging.WARNING),
            ('error', logging.ERROR),
            ('critical', logging.CRITICAL),
        ],
    )
    def test_methods_emit_at_expected_level(self, caplog, method, level):
        # Given
        log = Logger()
        log.level = logging.DEBUG

        # When
        with caplog.at_level(logging.DEBUG, logger=PACKAGE_LOGGER_NAME):
            getattr(log, method)('hello %s', 'world')

        # Then
        records = [r for r in caplog.records if r.name == PACKAGE_LOGGER_NAME]
        assert len(records) == 1
        assert records[0].levelno == level
        assert records[0].getMessage() == 'hello world'

    def test_exception_logs_at_error_with_traceback(self, caplog):
        # Given
        log = Logger()
        log.level = logging.DEBUG

        # When
        with caplog.at_level(logging.DEBUG, logger=PACKAGE_LOGGER_NAME):
            try:
                raise RuntimeError('boom')
            except RuntimeError:
                log.exception('caught it')

        # Then
        records = [r for r in caplog.records if r.name == PACKAGE_LOGGER_NAME]
        assert len(records) == 1
        assert records[0].levelno == logging.ERROR
        assert records[0].exc_info is not None


# ---------------------------------------------------------------------------
# Logger.at_level context manager
# ---------------------------------------------------------------------------


class TestAtLevel:
    def test_temporarily_sets_and_restores_level(self):
        # Given
        log = Logger()
        log.level = logging.WARNING

        # When
        with log.at_level(logging.ERROR):
            # Then inside the context the temporary level applies
            assert log.level == logging.ERROR

        # Then the previous level is restored on exit
        assert log.level == logging.WARNING

    def test_accepts_level_name(self):
        # Given
        log = Logger()
        log.level = logging.INFO

        # When
        with log.at_level('CRITICAL'):
            assert log.level == logging.CRITICAL

        # Then
        assert log.level == logging.INFO

    def test_restores_level_on_exception(self):
        # Given
        log = Logger()
        log.level = logging.WARNING

        # When an exception escapes the context
        with pytest.raises(RuntimeError):
            with log.at_level(logging.ERROR):
                raise RuntimeError('boom')

        # Then the previous level is still restored
        assert log.level == logging.WARNING


# ---------------------------------------------------------------------------
# suspend / resume shims
# ---------------------------------------------------------------------------


class TestSuspendResume:
    def test_suspend_silences_then_resume_restores(self):
        # Given
        log = Logger()
        log.level = logging.INFO

        # When
        log.suspend()

        # Then nothing below CRITICAL gets through
        assert log.level > logging.CRITICAL

        # When
        log.resume()

        # Then the pre-suspend level is restored
        assert log.level == logging.INFO

    def test_resume_uses_level_captured_at_suspend(self):
        # Given a level set before suspending
        log = Logger()
        log.level = logging.ERROR
        log.suspend()

        # When
        log.resume()

        # Then
        assert log.level == logging.ERROR
