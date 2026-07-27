"""
Works around a pre-existing name collision that breaks pytest collection
in this repo: the repo root (this file's directory) is itself named
"mbt_gym" AND has an __init__.py (both pre-existing, not added by this
change), and it also contains a real, separate `mbt_gym` LIBRARY package
one level down (<repo_root>/mbt_gym/) -- a leftover of how the upstream
mbt_gym repo was cloned in.

Whenever pytest needs to resolve the dotted import name of a collected
test file that lives (directly or via a subpackage) under the repo root,
its default "prepend" import mode walks up through __init__.py-bearing
ancestors and, on the way, imports the repo root itself as the top-level
module "mbt_gym" (since that's its directory name) -- colliding with, and
silently shadowing, the real mbt_gym library under the exact same name.
This can happen freshly for EACH collected test file (pytest's rootpath
insertion is not a one-time effect), so a one-time fix in module-level
code or even in pytest_configure is not reliably enough; the correct
module is force-reasserted into sys.modules before every collection
step via pytest_collectstart.

This is a pytest-collection workaround only. It does not change any
production import, file, or package layout -- from any script's own
`import mbt_gym...`, the real library is already found correctly by cwd-
based sys.path resolution ("python script.py" from the repo root), which
does not exhibit this problem at all.
"""

import sys
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_REAL_MBT_GYM_INIT = REPO_ROOT / "mbt_gym" / "__init__.py"


def _load_real_mbt_gym_module():
    spec = importlib.util.spec_from_file_location(
        "mbt_gym", _REAL_MBT_GYM_INIT, submodule_search_locations=[str(_REAL_MBT_GYM_INIT.parent)]
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_real_mbt_gym_module = _load_real_mbt_gym_module()


def _reassert_real_mbt_gym():
    cached = sys.modules.get("mbt_gym")
    if cached is not _real_mbt_gym_module:
        sys.modules["mbt_gym"] = _real_mbt_gym_module


def pytest_configure(config):
    _reassert_real_mbt_gym()


def pytest_collectstart(collector):
    _reassert_real_mbt_gym()
