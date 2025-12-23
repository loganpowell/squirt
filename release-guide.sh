#!/usr/bin/env bash
# Quick reference for releasing sleuth

cat << 'EOF'
╔═══════════════════════════════════════════════════════════════╗
║                   SLEUTH RELEASE GUIDE                        ║
╚═══════════════════════════════════════════════════════════════╝

📦 QUICK RELEASE (Automated)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Verify everything is ready:
   $ ./verify-release.sh

2. Preview the release (dry run):
   $ ./release.sh patch --dry-run

3. Execute the release:
   $ ./release.sh patch      # Bug fixes:     0.1.0 → 0.1.1
   $ ./release.sh minor      # New features:  0.1.1 → 0.2.0
   $ ./release.sh major      # Breaking:      0.2.0 → 1.0.0

The script will:
  ✅ Validate git status
  ✅ Bump version
  ✅ Update CHANGELOG.md
  ✅ Run tests
  ✅ Build package
  ✅ Create git tag
  ✅ Push to GitHub
  ✅ Publish to PyPI (with confirmation)

📋 PRE-RELEASE CHECKLIST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before running ./release.sh:

  □ All tests passing
  □ Documentation updated
  □ CHANGELOG.md has [Unreleased] section with changes
  □ Working directory is clean (no uncommitted changes)
  □ On main branch
  □ Latest changes pulled from GitHub
  □ PyPI credentials configured (~/.pypirc or env vars)

🔧 MANUAL COMMANDS (If needed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Build only:
  $ python -m build --clean

Validate package:
  $ python -m twine check dist/*

Test on TestPyPI:
  $ python -m twine upload --repository testpypi dist/*

Upload to PyPI:
  $ python -m twine upload dist/*

Create tag:
  $ git tag -a v0.2.0 -m "Release version 0.2.0"
  $ git push origin v0.2.0

🆘 TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Working directory is not clean":
  $ git status
  $ git add -A && git commit -m "commit message"

"Must be on main branch":
  $ git checkout main
  $ git pull origin main

"Version already exists on PyPI":
  Delete tag and bump to new version:
  $ git tag -d v0.2.0
  $ git push origin :refs/tags/v0.2.0

Build failures:
  $ rm -rf dist/ build/ *.egg-info
  $ python -m build

Missing build tools:
  $ pip install build twine

📚 DOCUMENTATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Full documentation:
  - RELEASE.md         Complete release process
  - CONTRIBUTING.md    Development guidelines
  - CHANGELOG.md       Version history

🔗 LINKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PyPI:    https://pypi.org/project/squirt/
GitHub:  https://github.com/loganpowell/squirt
Issues:  https://github.com/loganpowell/squirt/issues

EOF
