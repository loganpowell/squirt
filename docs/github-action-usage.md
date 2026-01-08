# Example: Using Squirt Metrics Action

## Simple Usage

```yaml
name: Metrics Report

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

permissions:
  contents: write # For committing history
  pull-requests: write # For PR comments

jobs:
  metrics:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      # Run your tests with squirt instrumentation
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest tests/

      # Generate metrics report (uses squirt config for paths)
      - name: Generate Metrics Report
        uses: loganpowell/squirt@v1
```

## With Configuration

Squirt automatically uses your configured directories and settings:

```python
# In your code or conftest.py
from squirt import configure

configure(
    results_dir="tests/results",
    history_dir="tests/history",
    regression_threshold=5.0
)
```

The action will read these settings automatically!

## Advanced Usage

```yaml
name: Metrics Report

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
  workflow_dispatch:

permissions:
  contents: write
  pull-requests: write

jobs:
  metrics:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run instrumented tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          pytest tests/ --squirt-enabled
        continue-on-error: true

      - name: Generate Metrics Report
        uses: loganpowell/squirt@v1
        with:
          commit-history: auto # Only commit on push events
          create-pr-comment: true
          regression-threshold: 10.0 # Override config threshold
          python-version: "3.12"
```

## Monorepo Usage

```yaml
name: Backend Metrics

on:
  push:
    paths:
      - "backend/**"
    branches: [main]
  pull_request:
    paths:
      - "backend/**"
    branches: [main]

permissions:
  contents: write
  pull-requests: write

jobs:
  backend-metrics:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Run backend tests
        run: |
          cd backend
          pip install -r requirements.txt
          pytest tests/

      - name: Generate Backend Metrics
        uses: loganpowell/squirt@v1
        with:
          working-directory: backend
```

## Inputs

| Input                  | Description                                        | Default                |
| ---------------------- | -------------------------------------------------- | ---------------------- |
| `working-directory`    | Working directory                                  | `.`                    |
| `commit-history`       | Commit history to repo (`true`, `false`, `auto`)   | `auto`                 |
| `create-pr-comment`    | Create/update PR comment                           | `true`                 |
| `regression-threshold` | Accuracy regression threshold % (overrides config) | _(uses squirt config)_ |
| `github-token`         | GitHub token for PR comments                       | `${{ github.token }}`  |
| `python-version`       | Python version to use                              | `3.12`                 |

**Note:** `results-dir` and `history-dir` are automatically read from your squirt configuration!

## Outputs

| Output            | Description                              |
| ----------------- | ---------------------------------------- |
| `report-path`     | Path to the generated full report        |
| `pr-comment-path` | Path to the generated PR comment         |
| `has-regression`  | Whether accuracy regression was detected |

## Permissions Required

```yaml
permissions:
  contents: write # For committing history back to repo
  pull-requests: write # For creating/updating PR comments
```

## Notes

- History is automatically committed back to the repo on push events (not PRs)
- Uses `[skip ci]` to avoid triggering workflows
- Artifacts are uploaded for 30 days (reports) and 90 days (history)
- Regression check will fail the workflow if accuracy drops more than threshold
