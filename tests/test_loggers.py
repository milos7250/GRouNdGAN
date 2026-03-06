import logging

import pytest


def test_getLogger_no_error(caplog: pytest.LogCaptureFixture) -> None:
    import loggers  # pyright: ignore[reportUnusedImport]  # noqa: F401
    
    logger = logging.getLogger("getLogger")
    logger.warning("warning")
    
    assert all(record.levelno < logging.ERROR for record in caplog.records)

def test_setup_logger_no_error(caplog: pytest.LogCaptureFixture) -> None:
    from loggers import setup_logger
    
    logger = setup_logger("setup_logger")
    logger.warning("warning")
    
    assert all(record.levelno < logging.ERROR for record in caplog.records)

    
def test_getLogger_error(caplog: pytest.LogCaptureFixture) -> None:
    import loggers  # pyright: ignore[reportUnusedImport]  # noqa: F401
    
    logger = logging.getLogger("getLogger")
    logger.error("error")
    
    assert any(record.levelno >= logging.ERROR for record in caplog.records)

def test_setup_logger_error(caplog: pytest.LogCaptureFixture) -> None:
    from loggers import setup_logger
    
    logger = setup_logger("setup_logger")
    logger.error("error")
    
    assert any(record.levelno >= logging.ERROR for record in caplog.records)

    
    