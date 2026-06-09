import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "audit_runner_dependencies_for_test",
        SCRIPTS / "audit_runner_dependencies.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_audit_runner_detects_missing_tmp_dependency(tmp_path: Path):
    module = load_module()
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "main.py").write_text("from helper import x\nprint(x)\n", encoding="utf-8")
    (scripts / "helper.py").write_text("x = 1\n", encoding="utf-8")
    runner = scripts / "run.sh"
    runner.write_text(
        '\n'.join(
            [
                f'MAIN_SCRIPT="${{MAIN_SCRIPT:-{scripts / "main.py"}}}"',
                'scp "${MAIN_SCRIPT}" "${SERVER}:/tmp/main.py"',
                'ssh "${SERVER}" "python3 /tmp/main.py"',
            ]
        ),
        encoding="utf-8",
    )

    result = module.audit_runner(runner, scripts)

    assert result["issues"] == [
        {
            "script": "main.py",
            "dependency": "helper.py",
            "error": "missing_uploaded_local_dependency",
        }
    ]


def test_audit_current_runners_have_uploaded_tmp_dependencies():
    module = load_module()
    report = module.audit(SCRIPTS)

    assert report["issue_count"] == 0
