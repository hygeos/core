import pytest

from core.tools import reglob

@pytest.mark.parametrize('regexp', ['.*.py','[a-z]+.*'])
def test_reglob(regexp):
    p = 'core'
    l = reglob(p, regexp)
    assert len(l) != 0