import qme


def test_version_and_deps_exposed():
    assert isinstance(qme.__version__, str)
    assert hasattr(qme, "deps")


def test_mock_calculator_available():
    calc = qme.MockCalculator(backend="mock")
    assert calc is not None
