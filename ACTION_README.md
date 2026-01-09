# Squirt GitHub Action 🚀

Generate comprehensive metrics reports and detect regressions automatically in your CI/CD pipeline.

[![GitHub Marketplace](https://img.shields.io/badge/Marketplace-Squirt%20Metrics-blue?logo=github)](https://github.com/marketplace/actions/squirt-metrics-report)

## Features

- 📊 **Comprehensive Reports** - Full metrics analysis with historical trends
- 📈 **Regression Detection** - Automatic accuracy regression checks with configurable thresholds
- 💬 **PR Comments** - Beautiful, actionable metrics summaries on every PR
- 📝 **Job Summaries** - Quick overview in GitHub Actions UI
- 🔄 **History Tracking** - Commits metrics history back to your repo
- 🎯 **Flexible Filtering** - Skip or include specific metric namespaces via commit messages
- ⚙️ **Auto-Configuration** - Uses your existing squirt config

## Quick Start

### Minimal Setup

```yaml
name: Metrics Report

on: [push, pull_request]

permissions:
  contents: write # For committing history
  pull-requests: write # For PR comments

jobs:
  metrics:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0 # Required for history

      - name: Generate Metrics Report
        uses: loganpowell/squirt@v1
        with:
          working-directory: backend # Optional, default: .
          python-version: "3.12" # Optional, default: 3.12
```

That's it! The action automatically:

- ✅ Sets up Python and installs dependencies
- ✅ Runs your instrumented tests
- ✅ Generates full report and PR comments
- ✅ Commits history back to repo

## Commit Message Filtering 🎯

Control which metrics run using commit message flags:

```bash
# Skip specific namespaces
git commit -m "Fix pipeline [skip-metrics-namespaces:azure,echo]"

# Only run specific namespaces
git commit -m "Test core only [only-metrics-namespaces:m,tokens]"

# Multiple flags
git commit -m "Update tests [skip-metrics-namespaces:azure] [only-metrics-namespaces:m]"
```

The action automatically parses these flags and passes them to pytest.

## Configuration Options

### Inputs

| Input                  | Description                                 | Default               |
| ---------------------- | ------------------------------------------- | --------------------- |
| `working-directory`    | Directory containing your tests             | `.`                   |
| `python-version`       | Python version to use                       | `3.12`                |
| `run-tests`            | Whether to run tests                        | `true`                |
| `test-paths`           | Test paths to run                           | `tests/instrumented/` |
| `test-maxfail`         | Stop after N failures                       | (none)                |
| `extra-pytest-args`    | Additional pytest args                      | (none)                |
| `commit-history`       | Commit history back (`true`/`false`/`auto`) | `auto`                |
| `create-pr-comment`    | Create PR comments                          | `true`                |
| `regression-threshold` | Accuracy regression threshold %             | (uses squirt config)  |
| `github-token`         | GitHub token                                | `${{ github.token }}` |

### Outputs

| Output            | Description                                  |
| ----------------- | -------------------------------------------- |
| `report-path`     | Path to full metrics report                  |
| `pr-comment-path` | Path to PR comment markdown                  |
| `has-regression`  | Whether regression detected (`true`/`false`) |

## Examples

### With Custom Configuration

```yaml
- name: Generate Metrics Report
  uses: loganpowell/squirt@v1
  with:
    working-directory: backend
    test-paths: tests/unit/ tests/integration/
    test-maxfail: 3
    regression-threshold: 5.0
    python-version: "3.11"
```

### With Environment Secrets

```yaml
- name: Generate Metrics Report
  uses: loganpowell/squirt@v1
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    AZURE_OPENAI_ENDPOINT: ${{ secrets.AZURE_OPENAI_ENDPOINT }}
```

### Skip History Commit on PRs

```yaml
- name: Generate Metrics Report
  uses: loganpowell/squirt@v1
  with:
    commit-history: ${{ github.event_name == 'push' }}
```

### With Namespace Filtering

```yaml
- name: Parse commit flags
  id: flags
  run: |
    SKIP_NS=$(echo "${{ github.event.head_commit.message }}" | grep -oP '\[skip-metrics-namespaces:\K[^\]]+' || echo "")
    echo "skip=$SKIP_NS" >> $GITHUB_OUTPUT

- name: Generate Metrics Report
  uses: loganpowell/squirt@v1
  with:
    extra-pytest-args: "--skip-metrics-namespaces=${{ steps.flags.outputs.skip }}"
```

## What Gets Generated

### 1. Full Report (artifact)

Comprehensive metrics with:

- Component-level breakdown
- Historical trends (sparklines)
- System metric aggregation
- Detailed timing and accuracy data

### 2. PR Comment

Concise summary showing:

- Key metrics overview
- Comparison to main branch
- Regression warnings (if any)
- Trend indicators

### 3. Job Summary

Quick view in GitHub Actions UI with:

- Pass/fail status
- Critical metrics
- Links to full report

## Repository Setup

### Required Permissions

Your workflow needs:

```yaml
permissions:
  contents: write # To commit history
  pull-requests: write # To create PR comments
```

### Directory Structure

The action expects (and auto-creates):

```
your-repo/
├── tests/
│   ├── instrumented/     # Your @track decorated tests
│   ├── results/          # Generated by squirt (gitignored)
│   └── history/          # Committed to track trends
├── pyproject.toml        # Or requirements.txt
└── .github/
    └── workflows/
        └── metrics.yml   # Your workflow
```

### Gitignore

Add to `.gitignore`:

```gitignore
# Squirt generates these
tests/results/
*.squirt.json
```

Do **not** ignore `tests/history/` - it needs to be committed!

## Troubleshooting

### Action fails with "No tests found"

Ensure your `test-paths` input points to tests with `@track` decorators:

```yaml
with:
  test-paths: tests/instrumented/ # Adjust to your path
```

### PR comments not appearing

Check workflow permissions:

```yaml
permissions:
  pull-requests: write # Required!
```

### History not committing

Ensure `fetch-depth: 0` in checkout:

```yaml
- uses: actions/checkout@v4
  with:
    fetch-depth: 0 # Required for git history
```

### Regression checks failing unexpectedly

Adjust threshold or check your squirt config:

```yaml
with:
  regression-threshold: 10.0 # Increase tolerance
```

## Advanced: Monorepo Setup

```yaml
strategy:
  matrix:
    service: [auth, payments, notifications]

steps:
  - uses: actions/checkout@v4
    with:
      fetch-depth: 0

  - name: Generate Metrics for ${{ matrix.service }}
    uses: loganpowell/squirt@v1
    with:
      working-directory: services/${{ matrix.service }}
      test-paths: tests/
```

## Documentation

- **[Squirt Library Docs](https://github.com/loganpowell/squirt/blob/main/README.md)** - How to instrument your code
- **[Custom Metrics Guide](https://github.com/loganpowell/squirt/blob/main/squirt/docs/custom_metrics_guide.md)** - Create domain-specific metrics
- **[API Analysis](https://github.com/loganpowell/squirt/blob/main/squirt/docs/API_ANALYSIS.md)** - Understand the metric patterns

## Support

- 🐛 [Report an issue](https://github.com/loganpowell/squirt/issues)
- 💬 [Discussions](https://github.com/loganpowell/squirt/discussions)
- 📖 [Full documentation](https://github.com/loganpowell/squirt)

## License

MIT
