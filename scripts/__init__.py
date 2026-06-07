# Makes `from scripts.x import y` work when pytest is run from the repo root.
# Required by tests/test_environment.py (test_all_stub_files_exist) and every test
# that imports a script module directly.
