import hpfem


def test_version_is_semver():
    parts = hpfem.__version__.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts)
