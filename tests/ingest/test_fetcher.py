import subprocess

import pytest

from fastapi_rag_lab.ingest.fetcher import fetch_fastapi_docs


def _network_available() -> bool:
    try:
        result = subprocess.run(
            ["git", "ls-remote", "https://github.com/fastapi/fastapi.git", "HEAD"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


network = pytest.mark.skipif(not _network_available(), reason="Network unavailable")


@pytest.mark.network
@network
def test_fetch_clones_and_extracts_markdown(tmp_path):
    clone_dir = tmp_path / "fastapi"
    docs_dir = tmp_path / "fastapi_docs"

    markdown_paths = fetch_fastapi_docs(
        clone_dir=clone_dir,
        docs_dir=docs_dir,
    )

    assert len(markdown_paths) > 0
    assert all(p.suffix == ".md" for p in markdown_paths)
    assert all(p.is_file() for p in markdown_paths)
    assert docs_dir.exists()


def test_fetch_rejects_nonexistent_sha(tmp_path):
    clone_dir = tmp_path / "fastapi"
    docs_dir = tmp_path / "fastapi_docs"

    with pytest.raises(RuntimeError, match="git clone failed|Could not fetch"):
        fetch_fastapi_docs(
            clone_dir=clone_dir,
            docs_dir=docs_dir,
            commit_sha="0" * 40,
        )


def test_fetch_rejects_sha_mismatch(tmp_path):
    clone_dir = tmp_path / "fastapi"
    clone_dir.mkdir()

    subprocess.run(["git", "init", str(clone_dir)], capture_output=True, check=True)
    (clone_dir / "dummy.txt").write_text("x")
    subprocess.run(
        ["git", "-C", str(clone_dir), "add", "."], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(clone_dir), "commit", "-m", "init"],
        capture_output=True,
        check=True,
        env={
            "GIT_AUTHOR_NAME": "test",
            "GIT_COMMITTER_NAME": "test",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "HOME": str(tmp_path),
            "PATH": "/usr/bin:/bin:/usr/local/bin",
        },
    )

    with pytest.raises(RuntimeError, match="expected"):
        fetch_fastapi_docs(
            clone_dir=clone_dir,
            docs_dir=tmp_path / "docs",
            commit_sha="a" * 40,
        )
