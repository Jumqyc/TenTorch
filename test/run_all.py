"""Run every check in this directory, one process at a time.

The checks are the executable specification of ``mpsdmrg``: each one asserts its contract and
prints a final line.  The runner reports that line and a non-zero exit status for a failure.
Args:
  None.
Returns:
  The process exit status: 0 when every check passed, 1 otherwise.
"""
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = HERE / '.cache'


def main() -> int:
    '''Run ``check_*.py`` serially and return the exit status of the run.'''
    CACHE.mkdir(exist_ok = True)
    failures = 0
    for script in sorted(HERE.glob('check_*.py')):
        failed = False
        try:
            completed = subprocess.run(
                [sys.executable, '-u', str(script)],
                cwd = ROOT,
                capture_output = True,
                text = True,
                timeout = 3600,
                env = {
                    **os.environ,
                    'PYTHONPATH': str(ROOT),
                    'TRITON_CACHE_DIR': str(CACHE),
                })
            lines = completed.stdout.strip().splitlines()
            last = lines[-1] if lines else completed.stderr.strip().splitlines()[-1]
            failed = completed.returncode != 0
        except subprocess.TimeoutExpired:
            last = 'timeout after 3600 s'
            failed = True
        if failed:
            failures += 1
        print(f'{script.name:26s} {last:64s} rc={"fail" if failed else "ok"}')
    print(f'{failures} failure(s)')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
