import ast
from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _load_example_users() -> dict[int, str]:
    """Extract EXAMPLE_USERS via AST instead of importing app.py.

    Importing app.py directly triggers Streamlit calls and a dataset
    download at module level, so we parse the literal dict out of the
    source instead of executing it.
    """
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "EXAMPLE_USERS" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("EXAMPLE_USERS assignment not found in app.py")


def test_example_users_has_50_entries():
    assert len(_load_example_users()) == 50


def test_example_users_ids_are_unique():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "EXAMPLE_USERS" for target in node.targets
        ):
            assert len(node.value.keys) == len(ast.literal_eval(node.value))
            return
    raise AssertionError("EXAMPLE_USERS assignment not found in app.py")
