def pytest_configure(config):
    # Registered here rather than in the shared pytest.ini; deselect with -m "not slow".
    config.addinivalue_line("markers", "slow: runs the whole pipeline on the small dataset (minutes)")
