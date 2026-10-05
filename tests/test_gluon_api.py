import sys
from types import ModuleType, SimpleNamespace

import pytest

from gfaas.triton_compat import ast_source_type


@pytest.mark.parametrize("public", [False, True])
def test_gluon_ast_source_probes_public_then_legacy(monkeypatch, public):
    gluon = ModuleType("triton.experimental.gluon")
    experimental = ModuleType("triton.experimental")
    experimental.gluon = gluon
    legacy = ModuleType("triton.experimental.gluon._runtime")
    legacy.GluonASTSource = object()
    expected = object() if public else legacy.GluonASTSource
    if public:
        gluon.GluonASTSource = expected
    for module in (gluon, experimental, legacy):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    assert ast_source_type(SimpleNamespace(is_gluon=lambda: True)) is expected


z_gluon = ModuleType("triton.experimental.gluon.language")
a_layout = type("Layout", (), {"__module__": "triton.experimental.gluon.language"})


def source_order_kernel(X):
    return z_gluon.arange(X, layout=a_layout())


def test_source_bundle_is_stable_across_python_hash_seeds():
    import os
    import subprocess
    from pathlib import Path

    script = """
from types import SimpleNamespace
from test_gluon_api import source_order_kernel
from gfaas import triton_compat as compat
class Jit:
    fn = staticmethod(source_order_kernel)
compat._types = lambda frontend: (Jit, object)
compat.validate_kernel = lambda kernel, frontend: (kernel, None)
print(compat.source_bundle(Jit(), 'gluon'))
"""
    root = Path(__file__).resolve().parents[1]
    outputs = []
    for seed in ("1", "2", "3", "42"):
        environment = dict(
            os.environ,
            PYTHONHASHSEED=seed,
            PYTHONPATH=str(root / "tests") + os.pathsep + str(root / "src"),
        )
        outputs.append(subprocess.check_output([sys.executable, "-c", script], env=environment))
    assert all(value == outputs[0] for value in outputs)
