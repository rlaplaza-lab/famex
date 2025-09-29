def test_debug_qme_import():
    import importlib.util
    import inspect
    import sys

    import qme

    print("qme.__file__=", getattr(qme, "__file__", None))
    print("qme.__dict__ keys sample=", sorted(list(qme.__dict__.keys())))
    print("sys.path[0:5]=", sys.path[:5])
    spec = importlib.util.find_spec("qme")
    print('importlib.util.find_spec("qme")=', spec)
    try:
        print("has QMEOptimizer:", hasattr(qme, "QMEOptimizer"))
        print("QMEOptimizer repr:", repr(getattr(qme, "QMEOptimizer", None)))
    except Exception as e:
        print("getattr QMEOptimizer raised", e)
    # Keep test lightweight
    assert True
