# Controlling Log Output

EasyScience uses Python's standard `logging` module for all
informational and warning messages. This gives you full control over
what output the library produces and where it goes.

## Quick Start

The most common operation is suppressing all EasyScience messages. You
can do it in one line:

```python
from easyscience import global_object

global_object.log.set_level('ERROR')
```

Or using the standard library directly:

```python
import logging

logging.getLogger('easyscience').setLevel(logging.ERROR)
```

Both forms set the package-root logger to `ERROR`, which suppresses all
`WARNING`, `INFO`, and `DEBUG` messages from EasyScience.

## Logger Hierarchy

EasyScience loggers are named hierarchically under the root
`easyscience` logger. This lets you suppress the whole package or
individual subsystems:

```
easyscience                        # root — controls everything
├── easyscience.base_classes       # EasyList runtime warnings
├── easyscience.legacy             # ObjBase / CollectionBase deprecations
├── easyscience.deprecated         # @deprecated decorator messages
├── easyscience.fitting            # import-availability warnings
│   ├── easyscience.fitting.bumps  # Bumps fitting runtime messages
│   ├── easyscience.fitting.lmfit  # LMFit fitting runtime messages
│   └── easyscience.fitting.dfo    # DFO fitting runtime messages
├── easyscience.variable           # parameter warnings
└── easyscience.global_object      # undo/redo, debugging diagnostics
```

Setting the level on a parent logger affects all its children, because
child loggers inherit the parent's effective level.

```python
import logging

# Suppress everything from the fitting subsystem (including bumps/lmfit/dfo)
logging.getLogger('easyscience.fitting').setLevel(logging.ERROR)

# Suppress only Bumps runtime messages
logging.getLogger('easyscience.fitting.bumps').setLevel(logging.ERROR)

# Suppress only legacy deprecation notices
logging.getLogger('easyscience.legacy').setLevel(logging.ERROR)
```

## Environment Variable

You can control the log level **before** importing EasyScience by
setting the `EASYSCIENCE_LOG_LEVEL` environment variable. This is useful
when a downstream library cannot configure logging in code before
EasyScience is imported.

**Accepted values:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`, or
a numeric level (e.g. `40`).

**Example — suppressing all import-time messages:**

```powershell
# Windows PowerShell
$env:EASYSCIENCE_LOG_LEVEL = 'ERROR'
python -c "import easyscience"  # silent
```

```bash
# Linux / macOS
EASYSCIENCE_LOG_LEVEL=ERROR python -c "import easyscience"  # silent
```

Or from Python, set the environment variable before the import:

```python
import os

os.environ['EASYSCIENCE_LOG_LEVEL'] = 'ERROR'
import easyscience  # now silent
```

## Temporary Suppression (Context Manager)

When you only want to silence messages during a specific operation, use
the `at_level` context manager:

```python
from easyscience import global_object
import logging

# Messages during the fit are suppressed, then the previous level is restored
with global_object.log.at_level(logging.ERROR):
    results = fitter.fit(x, y, weights)
```

This is the recommended pattern for technique-specific libraries that
don't want EasyScience output to leak into their own users' consoles.

## Convenience API

The `global_object.log` object exposes convenience methods that mirror
`logging` module-level functions:

| Method              | Description                                                          |
| ------------------- | -------------------------------------------------------------------- |
| `.set_level(level)` | Set the package-root logger level (`'WARNING'` or `logging.WARNING`) |
| `.debug(msg)`       | Log a `DEBUG`-level message                                          |
| `.info(msg)`        | Log an `INFO`-level message                                          |
| `.warning(msg)`     | Log a `WARNING`-level message                                        |
| `.error(msg)`       | Log an `ERROR`-level message                                         |
| `.critical(msg)`    | Log a `CRITICAL`-level message                                       |
| `.exception(msg)`   | Log an `ERROR`-level message with traceback                          |
| `.getLogger(name)`  | Get a child logger under `easyscience`                               |
| `.at_level(level)`  | Context manager for temporary level change                           |
| `.suspend()`        | Suppress all EasyScience output                                      |
| `.resume()`         | Restore the previously set level                                     |

## Recipes

### Recipe 1: Keep your test output clean

When running tests that exercise EasyScience fitting, suppress the
library's messages so they don't clutter your test reports:

```python
import logging
from easyscience import global_object

@pytest.fixture(autouse=True)
def _quiet_easyscience():
    with global_object.log.at_level(logging.ERROR):
        yield
```

### Recipe 2: Library author embedding EasyScience

If you maintain a library that uses EasyScience internally, prevent
EasyScience from writing to your users' consoles:

```python
# In your library's __init__.py, before any EasyScience import:
import os
os.environ.setdefault('EASYSCIENCE_LOG_LEVEL', 'ERROR')
```

### Recipe 3: Debug mode for development

When developing or debugging, see all EasyScience internal diagnostics:

```python
from easyscience import global_object
import logging

global_object.log.set_level('DEBUG')

# Your code here — all EasyScience messages will be visible
```

### Recipe 4: See only error messages

Keep the console clean but still see critical problems:

```python
from easyscience import global_object

global_object.log.set_level('ERROR')
```

## Library-Safe Behaviour

EasyScience follows standard library-logging best practices:

- **No `logging.basicConfig()`** — EasyScience never reconfigures the
  global logging setup.
- **No default stream handlers** — EasyScience does not attach handlers
  that write to `stdout` or `stderr`. It only creates log records.
  Applications and test frameworks decide where those records go.
- **Child loggers inherit** — Child loggers (e.g.
  `easyscience.fitting.bumps`) are left at `logging.NOTSET` by default,
  so they inherit level and handler configuration from the `easyscience`
  package-root logger. Supressing the root suppresses everything.

This means you are in full control. If you don't configure any handlers,
EasyScience messages will not appear. If you do configure handlers (as
`pytest` does via its logging plugin, or as an application might via
`basicConfig`), you control what is shown and where.
