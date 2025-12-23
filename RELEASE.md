# Release Process

This document describes how to release a new version of Squirt to PyPI.

## Prerequisites

1. **GitHub Repository**: Must be on `main` branch with clean working directory
2. **PyPI Account**: Need credentials configured in `~/.pypirc` or via environment variables
3. **Build Tools**: Requires `build` and `twine` packages

```bash
pip install build twine
```

## Quick Release

Use the automated release script:

```bash
# First, verify everything is ready
./verify-release.sh

# Then release
# Patch release (0.1.0 → 0.1.1)
./release.sh patch

# Minor release (0.1.1 → 0.2.0)
./release.sh minor

# Major release (0.2.0 → 1.0.0)
./release.sh major

# Dry run to preview changes
./release.sh patch --dry-run
```

## What the Script Does

The `release.sh` script automates the entire release process:

1. ✅ **Validates** git status (clean working directory, on main branch)
2. 📈 **Bumps** version in `pyproject.toml`
3. 📝 **Updates** `CHANGELOG.md` with release date
4. 💾 **Commits** version bump changes
5. 🏷️ **Creates** git tag `v<version>`
6. 🔨 **Builds** source distribution and wheel
7. ✔️ **Validates** package with `twine check`
8. ⬆️ **Pushes** commits and tags to GitHub
9. 🚀 **Publishes** to PyPI (with confirmation prompt)

## Manual Release

If you prefer to release manually:

### 1. Update Version

Edit `pyproject.toml`:

```toml
[project]
name = "squirt"
version = "0.2.0"  # Update this
```

### 2. Update Changelog

Edit `CHANGELOG.md`:

```markdown
## [Unreleased]

## [0.2.0] - 2025-12-23 # Add this line
```

### 3. Commit and Tag

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to 0.2.0"
git tag -a v0.2.0 -m "Release version 0.2.0"
```

### 4. Build Package

```bash
python -m build --clean
```

### 5. Check Distribution

```bash
python -m twine check dist/*
```

### 6. Push to GitHub

```bash
git push origin main
git push origin v0.2.0
```

### 7. Upload to PyPI

```bash
# Test on TestPyPI first (optional)
python -m twine upload --repository testpypi dist/*

# Upload to production PyPI
python -m twine upload dist/*
```

## PyPI Credentials

Configure credentials in `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-...your-token...

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-...your-token...
```

Or use environment variables:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...your-token...
```

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features (backward compatible)
- **Patch** (0.0.X): Bug fixes (backward compatible)

## Pre-release Checklist

Before running `./release.sh`:

- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated with changes
- [ ] No uncommitted changes
- [ ] On `main` branch
- [ ] Pulled latest from GitHub

## Post-release

After successful release:

1. Verify on PyPI: https://pypi.org/project/squirt/
2. Test installation: `pip install squirt==<version>`
3. Create GitHub release with notes from CHANGELOG
4. Announce on relevant channels

## Troubleshooting

### "Working directory is not clean"

```bash
git status
git add -A
git commit -m "your commit message"
```

### "Must be on main branch"

```bash
git checkout main
git pull origin main
```

### "Version already exists on PyPI"

Delete the git tag and start over:

```bash
git tag -d v0.2.0
git push origin :refs/tags/v0.2.0
```

Then bump to a new version.

### Build failures

Clean build artifacts:

```bash
rm -rf dist/ build/ *.egg-info
python -m build
```

### Upload failures

Check your credentials in `~/.pypirc` or environment variables.

## Emergency Rollback

If a bad release goes out:

1. **Yank the release** on PyPI (doesn't delete, marks as unavailable)
2. Fix the issues
3. Release a new patch version
4. Never delete releases from PyPI

```bash
# Yank via PyPI web interface
# Go to: https://pypi.org/manage/project/squirt/releases/
```
