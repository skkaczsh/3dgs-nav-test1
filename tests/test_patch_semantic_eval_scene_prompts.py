import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "patch_semantic_eval_scene_prompts_for_test",
        SCRIPTS / "patch_semantic_eval_scene_prompts.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_patch_text_replaces_single_prompt_and_preserves_items_schema():
    module = load_module()
    source = '''PROMPT = """
old prompt
""".strip()

def keep():
    return 1
'''
    updated, count = module.patch_text(source, module.REVIEW_PROMPT)

    assert count == 1
    assert "rooftop MANIFOLD/Mid360 scan" in updated
    assert '{"items":[{"mask_id":"1","label":"floor","confidence":0.90}]}' in updated
    assert "def keep()" in updated


def test_patch_file_creates_backup_once(tmp_path: Path):
    module = load_module()
    path = tmp_path / "review_merged_labels_prompt_v2.py"
    original = 'PROMPT = """\nold prompt\n""".strip()\n'
    path.write_text(original, encoding="utf-8")

    first = module.patch_file(path, module.REVIEW_PROMPT, dry_run=False)
    second = module.patch_file(path, module.REVIEW_PROMPT, dry_run=False)

    assert first["patched"] is True
    assert first["changed"] is True
    assert second["patched"] is True
    assert second["changed"] is False
    assert path.with_suffix(path.suffix + ".scene_prompt_bak").read_text(encoding="utf-8") == original
