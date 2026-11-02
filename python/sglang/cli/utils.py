import logging
import os
import subprocess
from functools import lru_cache

logger = logging.getLogger(__name__)


def try_get_model_path(extra_argv) -> str | None:
    """Return a model path from command-line arguments when one is present."""

    model_path = None
    for i, arg in enumerate(extra_argv):
        if arg in ("--model-path", "--model"):
            if i + 1 < len(extra_argv):
                model_path = extra_argv[i + 1]
                break
        elif arg.startswith("--model-path=") or arg.startswith("--model="):
            model_path = arg.split("=", 1)[1]
            break

    return model_path


def get_model_path(extra_argv):
    # Find the model_path argument
    model_path = try_get_model_path(extra_argv)

    if model_path is None:
        # Fallback for --help or other cases where model-path is not provided
        if any(h in extra_argv for h in ["-h", "--help"]):
            raise Exception(
                "Usage: sglang serve --model-path <model-name-or-path> [additional-arguments]\n\n"
            )
        else:
            raise Exception(
                "Error: --model-path is required. Please provide the path to the model."
            )
    return model_path


@lru_cache(maxsize=1)
def get_git_commit_hash() -> str:
    try:
        commit_hash = os.environ.get("SGLANG_GIT_COMMIT")
        if not commit_hash:
            commit_hash = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .strip()
                .decode("utf-8")
            )
        _CACHED_COMMIT_HASH = commit_hash
        return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        _CACHED_COMMIT_HASH = "N/A"
        return "N/A"
