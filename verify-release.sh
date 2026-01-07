#!/usr/bin/env bash
# Pre-release verification script
# Checks that the package is ready for release

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

CHECKS_PASSED=0
CHECKS_FAILED=0

check() {
    if eval "$2"; then
        echo -e "${GREEN}✅ $1${NC}"
        ((CHECKS_PASSED++))
        return 0
    else
        echo -e "${RED}❌ $1${NC}"
        ((CHECKS_FAILED++))
        return 1
    fi
}

info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

echo "🔍 Squirt Pre-Release Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# File existence checks
info "Checking required files..."
check "pyproject.toml exists" "test -f pyproject.toml"
check "README.md exists" "test -f README.md"
check "LICENSE exists" "test -f LICENSE"
check "CHANGELOG.md exists" "test -f CHANGELOG.md"
check "release.sh exists and is executable" "test -x release.sh"
echo

# Content checks
info "Checking file content..."
check "pyproject.toml has name='squirt'" "grep -q 'name = \"squirt\"' pyproject.toml"
check "pyproject.toml has version" "grep -q '^version = ' pyproject.toml"
check "README.md mentions installation" "grep -q 'pip install squirt' README.md"
check "CHANGELOG.md has Unreleased section" "grep -q '\[Unreleased\]' CHANGELOG.md"
echo

# Git checks
info "Checking git status..."
check "In git repository" "git rev-parse --git-dir > /dev/null 2>&1"
check "Working directory is clean" "test -z \"\$(git status --porcelain)\""
echo

# Python environment checks
info "Checking Python environment..."
check "Python 3.11+ available" "python --version 2>&1 | grep -qE 'Python 3\.(1[1-9]|[2-9][0-9])'"
check "pip available" "command -v pip > /dev/null"
check "build package installed" "python -c 'import build' 2>/dev/null"
check "twine package installed" "python -c 'import twine' 2>/dev/null"
echo

# Import checks
info "Checking package imports..."
check "squirt imports successfully" "python -c 'import squirt' 2>/dev/null"
check "squirt.m imports" "python -c 'from squirt import m' 2>/dev/null"
check "squirt.track imports" "python -c 'from squirt import track' 2>/dev/null"
check "squirt.configure imports" "python -c 'from squirt import configure' 2>/dev/null"
echo

# Test checks
info "Checking tests..."
if command -v pytest > /dev/null 2>&1; then
    check "Tests exist" "test -d squirt/tests"
    check "Tests pass" "pytest squirt/tests/ -q > /dev/null 2>&1"
else
    echo -e "${YELLOW}⚠️  pytest not found, skipping test checks${NC}"
fi
echo

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}Passed: $CHECKS_PASSED${NC}"
echo -e "${RED}Failed: $CHECKS_FAILED${NC}"
echo

if [ $CHECKS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed! Ready to release.${NC}"
    echo
    echo "To release, run:"
    echo "  ./release.sh [patch|minor|major]"
    exit 0
else
    echo -e "${RED}❌ Some checks failed. Fix issues before releasing.${NC}"
    exit 1
fi
