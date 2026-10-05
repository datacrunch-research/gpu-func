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
