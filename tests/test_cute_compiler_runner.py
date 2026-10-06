from gfaas import cute_backend, cute_compiler_runner


def test_failed_compile_retains_candidate_location(monkeypatch):
    namespace = {}
    exec(compile(
        'def kernel_entry():\n    raise AttributeError("launch returned None")\n',
        'candidate.py', 'exec'), namespace)

    def compile_variant(**request):
        namespace['kernel_entry']()

    monkeypatch.setattr(cute_backend, 'compile_variant', compile_variant)
    row = cute_compiler_runner._compile_one({'variant': {'id': 'failed-variant'}})

    assert row['status'] == 'failed'
    assert row['id'] == 'failed-variant'
    assert row['diagnostics'] == 'AttributeError: launch returned None'
    assert 'candidate.py", line 2, in kernel_entry' in row['traceback']
    assert row['wall_seconds'] >= 0
