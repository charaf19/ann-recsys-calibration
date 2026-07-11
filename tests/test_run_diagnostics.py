"""Child-process diagnostics for run_revision_experiments.run (Phase 4).

A failing child must stream its output live AND have its trailing traceback
embedded in StepError so the run manifest records the real failure, not just
the command line.
"""
import sys

import pytest

from run_revision_experiments import run, StepError


def _child(tmp_path, body):
    script = tmp_path / "child.py"
    script.write_text(body, encoding="utf-8")
    return ["python", str(script)]


def test_successful_command_returns_and_streams(tmp_path, capfd):
    cmd = _child(tmp_path,
                 "import sys\n"
                 "print('hello-stdout')\n"
                 "print('hello-stderr', file=sys.stderr)\n")
    run(cmd)  # must not raise
    out = capfd.readouterr().out
    assert "hello-stdout" in out
    assert "hello-stderr" in out  # stderr merged into stdout stream


def test_failing_command_raises_steperror_with_tail(tmp_path, capfd):
    cmd = _child(tmp_path,
                 "import sys\n"
                 "print('progress line 1')\n"
                 "print('a traceback-ish detail', file=sys.stderr)\n"
                 "sys.exit(3)\n")
    with pytest.raises(StepError) as exc:
        run(cmd)
    msg = str(exc.value)
    assert "exit 3" in msg, "exit code must be reported"
    assert "child.py" in msg, "the command must be reported"
    assert "traceback-ish detail" in msg, "child output tail must be embedded"

    out = capfd.readouterr().out
    assert "progress line 1" in out, "output must be streamed live too"


def test_failing_command_embeds_real_traceback(tmp_path):
    cmd = _child(tmp_path, "raise RuntimeError('boom in child')\n")
    with pytest.raises(StepError) as exc:
        run(cmd)
    msg = str(exc.value)
    assert "RuntimeError" in msg and "boom in child" in msg, \
        "the actual Python traceback tail must survive into StepError"


def test_tail_is_bounded(tmp_path):
    # Child prints many lines; only the bounded tail is retained/inspected.
    cmd = _child(tmp_path,
                 "for i in range(1000):\n"
                 "    print(f'line {i}')\n"
                 "import sys; sys.exit(1)\n")
    with pytest.raises(StepError) as exc:
        run(cmd, tail_lines=50)
    msg = str(exc.value)
    assert "line 999" in msg, "the most recent lines must be kept"
    assert "line 0" not in msg, "old lines beyond the tail must be dropped"
