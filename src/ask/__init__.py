"""ask - a private, local CLI for question-answering over the user's own corpora."""

import logging

# Fail-open judge loops log at debug via ask.* loggers. A NullHandler keeps those
# (and any unexpected warnings) from printing tracebacks to the user's terminal.
# `ask --help`-time config or ASK_DEBUG can attach a real handler for diagnostics.
logging.getLogger("ask").addHandler(logging.NullHandler())

__version__ = "0.1.0"
