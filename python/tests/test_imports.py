import importlib, os, pkgutil, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import trademaster_ml


def test_every_module_compiles():
    """A syntax or indentation error in any module (e.g. the daily update) fails here, before it can break a refresh."""
    import py_compile
    root = os.path.dirname(trademaster_ml.__file__)
    for m in pkgutil.iter_modules([root]):
        py_compile.compile(os.path.join(root, m.name + ".py"), doraise=True)


if __name__ == "__main__":
    test_every_module_compiles()
    print("ok test_every_module_compiles")
