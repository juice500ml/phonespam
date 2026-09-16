"""Resolve the release level from a pull request's labels.

Shared by two workflows so the rule is defined once:

* ``release-label.yml`` runs it on every PR as a required check, so a PR
  cannot be merged without a version label.
* ``publish.yml`` runs it again after the merge to pick the level it passes
  to semantic-release.

Reads the PR's labels as a JSON array in ``$LABELS``. Writes ``force=<level>``
to ``$GITHUB_OUTPUT`` when one is found, and exits non-zero with a GitHub
error annotation when none is.
"""

import json
import os
import sys

# Accepted labels, largest first: a PR carrying several takes the largest.
# These must match the values semantic-release's `force` input accepts.
LEVELS = ("major", "minor", "patch")


def main() -> int:
    labels = json.loads(os.environ.get("LABELS") or "[]")
    present = {str(name).lower() for name in labels}
    found = [level for level in LEVELS if level in present]

    if not found:
        pr = os.environ.get("PR", "?")
        shown = ", ".join(map(str, labels)) or "(none)"
        print(
            f"::error::PR #{pr} has no release label. Add exactly one of: "
            f"{', '.join(LEVELS)}. Labels on this PR: {shown}"
        )
        return 1

    level = found[0]
    if len(found) > 1:
        print(f"::warning::PR #{os.environ.get('PR', '?')} carries {found}; using {level}.")

    print(f"release level: {level}")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as fh:
            fh.write(f"force={level}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
