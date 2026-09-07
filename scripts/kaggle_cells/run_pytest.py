import sys
from pathlib import Path
ROOT = Path("/kaggle/working/qwen4exp-fork")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pytest
PYTEST_CODE = pytest.main([str(ROOT / "tests" / "models" / "jax" / "test_qwen4_exp.py"), "-q", "-s"])
print("pytest exit code:", PYTEST_CODE)
Path("/kaggle/working/pytest_summary.json").write_text(__import__("json").dumps({"exit_code": PYTEST_CODE}))
assert PYTEST_CODE == 0, "FAIL: fork unit tests failed on the TPU VM"
print("UNIT TESTS PASSED on TPU VM")
