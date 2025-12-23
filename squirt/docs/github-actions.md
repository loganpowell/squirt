# GitHub Actions Integration

Squirt is designed for seamless CI/CD integration with GitHub Actions.

## Quick Start

Add this workflow to your repository:

```yaml
# .github/workflows/metrics-report.yml
name: Generate Metrics Report

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

permissions:
  contents: read
  pull-requests: write

jobs:
  metrics:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          pip install poetry
          cd backend && poetry install

      - name: Run tests
        run: |
          cd backend
          poetry run pytest tests/integration/ -v

      - name: Generate Report
        run: |
          cd backend
          poetry run squirt report full --save-history --output ../report.md

      - name: Add to Job Summary
        run: cat report.md >> $GITHUB_STEP_SUMMARY

      - name: Generate PR Comment
        if: github.event_name == 'pull_request'
        run: |
          cd backend
          poetry run squirt report pr --output ../pr-comment.md

      - name: Post PR Comment
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const comment = fs.readFileSync('pr-comment.md', 'utf8');

            const { data: comments } = await github.rest.issues.listComments({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
            });

            const existing = comments.find(c => 
              c.user.type === 'Bot' && c.body.includes('Metrics Summary')
            );

            if (existing) {
              await github.rest.issues.updateComment({
                owner: context.repo.owner,
                repo: context.repo.repo,
                comment_id: existing.id,
                body: comment
              });
            } else {
              await github.rest.issues.createComment({
                owner: context.repo.owner,
                repo: context.repo.repo,
                issue_number: context.issue.number,
                body: comment
              });
            }
```

## Complete Workflow

Here's a production-ready workflow with all features:

```yaml
name: Generate Metrics Report

on:
  push:
    branches:
      - main
      - "release-*"
  pull_request:
    branches:
      - main
    types: [opened, synchronize, reopened]
  workflow_dispatch:

permissions:
  contents: read
  pull-requests: write
  actions: read

jobs:
  metrics:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0 # Full history for git hash detection

      - name: Check if backend exists
        id: check_backend
        run: |
          if [ -d "backend/squirt" ]; then
            echo "exists=true" >> $GITHUB_OUTPUT
          else
            echo "exists=false" >> $GITHUB_OUTPUT
            echo "⚠️ Squirt not found. Skipping metrics." >> $GITHUB_STEP_SUMMARY
          fi

      - name: Set up Python
        if: steps.check_backend.outputs.exists == 'true'
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install uv
        if: steps.check_backend.outputs.exists == 'true'
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"

      - name: Install dependencies
        if: steps.check_backend.outputs.exists == 'true'
        run: |
          cd backend
          uv sync --all-extras

      - name: Run tests and generate metrics
        if: steps.check_backend.outputs.exists == 'true'
        run: |
          cd backend
          uv run pytest tests/integration/ --maxfail=1 -v
        continue-on-error: true

      - name: Generate Full Report
        if: steps.check_backend.outputs.exists == 'true'
        run: |
          cd backend
          uv run squirt --results-dir tests/results --history-dir tests/history \
            report full --save-history --output ../metrics-report.md

      - name: Generate PR Comment
        if: steps.check_backend.outputs.exists == 'true' && github.event_name == 'pull_request'
        run: |
          cd backend
          uv run squirt --results-dir tests/results --history-dir tests/history \
            report pr --output ../pr-comment.md

      - name: Upload Full Report
        if: steps.check_backend.outputs.exists == 'true'
        uses: actions/upload-artifact@v4
        with:
          name: metrics-report
          path: metrics-report.md
          retention-days: 30

      - name: Upload Historical Data
        if: steps.check_backend.outputs.exists == 'true'
        uses: actions/upload-artifact@v4
        with:
          name: metrics-history
          path: backend/tests/history/
          retention-days: 90

      - name: Post PR Comment
        if: steps.check_backend.outputs.exists == 'true' && github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          script: |
            const fs = require('fs');
            const commentBody = fs.readFileSync('pr-comment.md', 'utf8');

            const { data: comments } = await github.rest.issues.listComments({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
            });

            const botComment = comments.find(comment => 
              comment.user.type === 'Bot' && 
              comment.body.includes('Metrics Summary')
            );

            if (botComment) {
              await github.rest.issues.updateComment({
                owner: context.repo.owner,
                repo: context.repo.repo,
                comment_id: botComment.id,
                body: commentBody
              });
            } else {
              await github.rest.issues.createComment({
                owner: context.repo.owner,
                repo: context.repo.repo,
                issue_number: context.issue.number,
                body: commentBody
              });
            }

      - name: Add Job Summary
        if: always()
        run: |
          if [ -f metrics-report.md ]; then
            cat metrics-report.md >> $GITHUB_STEP_SUMMARY
          else
            echo "⚠️ Metrics report not generated" >> $GITHUB_STEP_SUMMARY
          fi
```

## Directory Structure

Ensure your project has this structure:

```
backend/
├── squirt/              # Squirt library
├── tests/
│   ├── integration/     # Integration tests (run by CI)
│   ├── results/         # Test results (generated)
│   │   ├── system_heartbeat.json
│   │   ├── hierarchical_report.json
│   │   └── *_latest.json
│   └── history/         # Historical data (persisted)
│       ├── metrics_history.jsonl
│       └── *.{commit}.json
└── pyproject.toml
```

## Environment Variables

Squirt respects these environment variables:

| Variable              | Description             | Default           |
| --------------------- | ----------------------- | ----------------- |
| `SQUIRT_RESULTS_DIR`  | Results directory       | `./tests/results` |
| `SQUIRT_HISTORY_DIR`  | History directory       | `./tests/history` |
| `GITHUB_STEP_SUMMARY` | GitHub job summary file | (set by GitHub)   |

## Secrets

For LLM-based tests, configure secrets:

```yaml
env:
  AZURE_OPENAI_API_KEY: ${{ secrets.AZURE_OPENAI_API_KEY }}
  AZURE_OPENAI_ENDPOINT: ${{ secrets.AZURE_OPENAI_ENDPOINT }}
```

## PR Comment Features

The PR comment includes:

### Status Indicators

```markdown
## 🔴 1 critical issue(s)

## 🟠 2 high priority issue(s)

## 🟡 3 issue(s) to review

## ✅ All Checks Passed
```

### Metrics Table with Deltas

```markdown
| Metric     | Current | Δ   |
| ---------- | ------- | --- |
| accuracy   | 95.0%   | ↑   |
| runtime_ms | 2.5s    | →   |
| memory_mb  | 512 MB  | ↓   |
```

### Collapsible Component Details

```markdown
<details>
<summary>📦 Component Breakdown (Top 5)</summary>

| Component | Runtime |
| --------- | ------- |
| extractor | 1.5s    |
| processor | 1.0s    |

</details>
```

## Job Summary Features

The full report in job summary includes:

- **Mermaid Diagrams**: Treemaps render natively in GitHub
- **Sparklines**: Unicode sparklines for trends
- **Dependency Trees**: ASCII art hierarchy

## Caching History

To persist history between runs:

### Option 1: Artifacts (Recommended)

```yaml
- name: Download Previous History
  uses: dawidd6/action-download-artifact@v3
  with:
    name: metrics-history
    path: backend/tests/history/
    if_no_artifact_found: ignore
    workflow_conclusion: success
  continue-on-error: true

- name: Upload Updated History
  uses: actions/upload-artifact@v4
  with:
    name: metrics-history
    path: backend/tests/history/
```

### Option 2: Git Branch

```yaml
- name: Fetch history from branch
  run: |
    git fetch origin metrics-data:metrics-data || true
    git checkout metrics-data -- backend/tests/history/ || true

- name: Push updated history
  if: github.ref == 'refs/heads/main'
  run: |
    git config user.name "GitHub Actions"
    git config user.email "actions@github.com"
    git checkout -B metrics-data
    git add backend/tests/history/
    git commit -m "Update metrics history" || true
    git push origin metrics-data --force
```

## Conditional Workflows

### Run on Label

```yaml
on:
  pull_request:
    types: [opened, synchronize, labeled]

jobs:
  metrics:
    if: contains(github.event.pull_request.labels.*.name, 'run-metrics')
```

### Skip on Certain Files

```yaml
on:
  push:
    paths:
      - "backend/**"
      - "!backend/docs/**"
      - "!**.md"
```

## Failure Handling

### Continue on Test Failure

```yaml
- name: Run tests
  run: poetry run pytest tests/integration/
  continue-on-error: true

- name: Generate Report
  if: always() # Run even if tests fail
  run: poetry run squirt report full --output report.md
```

### Report Failures in Comment

```yaml
- name: Check for critical issues
  id: check_critical
  run: |
    if grep -q "🔴" pr-comment.md; then
      echo "has_critical=true" >> $GITHUB_OUTPUT
    fi

- name: Fail on critical
  if: steps.check_critical.outputs.has_critical == 'true'
  run: exit 1
```

## Notifications

### Slack Notification

```yaml
- name: Notify Slack on Regression
  if: failure()
  uses: slackapi/slack-github-action@v1
  with:
    payload: |
      {
        "text": "⚠️ Metrics regression detected in ${{ github.repository }}"
      }
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

## Best Practices

### 1. Always Save History

```yaml
- run: squirt report full --save-history
```

### 2. Archive History Long-Term

```yaml
retention-days: 90 # Keep 3 months of history
```

### 3. Use PR Comments for Visibility

Developers see metrics without clicking through to logs.

### 4. Set Up Branch Protection

Require metrics workflow to pass before merge.

### 5. Monitor Trends

Check job summaries regularly for gradual regressions.

## Troubleshooting

### Report Not Generated

```yaml
- name: Debug
  run: |
    ls -la backend/tests/results/
    cat backend/tests/results/system_heartbeat.json || echo "No heartbeat"
```

### PR Comment Not Posted

Check permissions:

```yaml
permissions:
  pull-requests: write
```

### History Not Persisting

Ensure artifact upload/download is correct:

```yaml
- name: Check history
  run: ls -la backend/tests/history/
```

## Next Steps

- [Metrics Guide](metrics.md) - Available metrics
- [Reporting Guide](reporting.md) - Report customization
- [API Reference](api.md) - Complete API documentation
