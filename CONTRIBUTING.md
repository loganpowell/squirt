# Squirt Development

## Project Structure

```
sleuth/
├── __init__.py           # Main exports
├── cli.py                # CLI interface
├── metrics.py            # Built-in metrics
├── core/                 # Core functionality
│   ├── decorator.py      # @track decorator
│   ├── types.py          # Core types
│   └── resource.py       # Resource tracking
├── categories/           # System metric categories
│   └── system.py
├── plugins/              # Plugin system
│   └── base.py
├── contrib/              # Contributed metrics
│   ├── vector/
│   ├── llm/
│   ├── chunk/
│   └── data/
├── reporting/            # Report generation
│   ├── aggregation.py
│   ├── reporter.py
│   └── insights.py
└── tests/                # Unit tests
```

## Publishing

See [RELEASE.md](RELEASE.md) for the complete release process.

Quick release:

```bash
./release.sh patch  # For bug fixes
./release.sh minor  # For new features
./release.sh major  # For breaking changes
```

## Development Setup

```bash
# Clone the repository
git clone https://github.com/loganpowell/squirt.git
cd squirt

# Install in development mode with all dependencies
pip install -e ".[dev,all]"

# Or with uv
uv pip install -e ".[dev,all]"

# Or use make
make install
```

## Common Development Tasks

Use the Makefile for common workflows:

```bash
make help          # Show all available commands
make test          # Run tests
make test-cov      # Run tests with coverage
make lint          # Check code quality
make build         # Build package
make verify        # Verify ready for release
make guide         # Show release guide

# Quick release workflow
make verify && make release-patch
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sleuth --cov-report=html

# Run specific test file
pytest tests/test_metrics_unit.py -v

# Or use make
make test
make test-cov
```

## Building for PyPI

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# Check the distribution
twine check dist/*

# Upload to TestPyPI (for testing)
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

## Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check . --fix

# Type checking
mypy sleuth
```

## Publishing Checklist

- [ ] Update version in `pyproject.toml`
- [ ] Update `CHANGELOG.md`
- [ ] Run all tests: `pytest`
- [ ] Build package: `python -m build`
- [ ] Check distribution: `twine check dist/*`
- [ ] Test install: `pip install dist/*.whl`
- [ ] Tag release: `git tag v0.1.0 && git push --tags`
- [ ] Upload to PyPI: `twine upload dist/*`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Documentation

Documentation is in `docs/` and the main `README.md`. When adding features:

1. Update docstrings
2. Add examples to README
3. Update API documentation
4. Add to CHANGELOG
