# Squirt Metrics Action

A GitHub Action for generating and reporting metrics from squirt-instrumented tests.

## Features

- 📊 Generates comprehensive metrics reports
- 📈 Tracks performance trends across commits
- 💬 Automatic PR comments with key metrics
- 🔍 Regression detection with configurable thresholds
- 📦 Commits history automatically (configurable)
- 🎯 Works with monorepos
- ⚙️ Uses your existing squirt configuration

## Quick Start

```yaml
- name: Generate Metrics Report
  uses: loganpowell/squirt@v1
```

The action automatically reads your squirt configuration, so no additional setup is needed!

## Documentation

For detailed usage examples, configuration options, and advanced scenarios, see:

**[docs/github-action-usage.md](docs/github-action-usage.md)**

## License

MIT
