import subprocess
import glob

tests = glob.glob("baselines/tests/test_*.py")
for t in tests:
    print(f"Running {t}...")
    res = subprocess.run(["uv", "run", "pytest", t, "-v", "--tb=short"], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"FAILED: {t}")
        # print(res.stdout)
    else:
        print(f"PASSED: {t}")
