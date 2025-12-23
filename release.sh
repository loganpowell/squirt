#!/usr/bin/env bash
set -euo pipefail

# Squirt Release Script
# Usage: ./release.sh [major|minor|patch] [--dry-run]
#
# This script:
# 1. Validates git status (clean working directory, on main branch)
# 2. Bumps version in pyproject.toml
# 3. Updates CHANGELOG.md with release date
# 4. Commits version bump
# 5. Creates git tag
# 6. Builds package
# 7. Publishes to PyPI
# 8. Pushes commits and tags to GitHub

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYPROJECT="pyproject.toml"
CHANGELOG="CHANGELOG.md"
REQUIRED_BRANCH="main"

# Parse arguments
VERSION_PART="${1:-patch}"
DRY_RUN=false

if [[ "${2:-}" == "--dry-run" ]] || [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo -e "${YELLOW}🏃 DRY RUN MODE - No changes will be made${NC}"
fi

# Validate version part
if [[ ! "$VERSION_PART" =~ ^(major|minor|patch)$ ]]; then
    echo -e "${RED}❌ Invalid version part: $VERSION_PART${NC}"
    echo "Usage: $0 [major|minor|patch] [--dry-run]"
    exit 1
fi

# Helper functions
info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN] Would run: $*${NC}"
    else
        "$@"
    fi
}

# Check if running from sleuth directory
if [[ ! -f "$PYPROJECT" ]]; then
    error "Must run from sleuth directory (where pyproject.toml is located)"
fi

info "Starting release process for sleuth..."

# 1. Check git status
info "Checking git repository status..."

if ! git rev-parse --git-dir > /dev/null 2>&1; then
    error "Not in a git repository"
fi

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$CURRENT_BRANCH" != "$REQUIRED_BRANCH" ]]; then
    error "Must be on $REQUIRED_BRANCH branch (currently on $CURRENT_BRANCH)"
fi

if [[ -n $(git status --porcelain) ]]; then
    error "Working directory is not clean. Commit or stash changes first."
fi

# Pull latest changes
info "Pulling latest changes from origin..."
run_cmd git pull origin "$REQUIRED_BRANCH"

success "Git repository is clean and up to date"

# 2. Get current version and calculate new version
info "Calculating new version..."

CURRENT_VERSION=$(grep -m 1 '^version = ' "$PYPROJECT" | sed 's/version = "\(.*\)"/\1/')
info "Current version: $CURRENT_VERSION"

IFS='.' read -r -a version_parts <<< "$CURRENT_VERSION"
MAJOR="${version_parts[0]}"
MINOR="${version_parts[1]}"
PATCH="${version_parts[2]}"

case $VERSION_PART in
    major)
        MAJOR=$((MAJOR + 1))
        MINOR=0
        PATCH=0
        ;;
    minor)
        MINOR=$((MINOR + 1))
        PATCH=0
        ;;
    patch)
        PATCH=$((PATCH + 1))
        ;;
esac

NEW_VERSION="${MAJOR}.${MINOR}.${PATCH}"
success "New version: $NEW_VERSION"

# 3. Update pyproject.toml
info "Updating $PYPROJECT..."
if [ "$DRY_RUN" = false ]; then
    sed -i.bak "s/^version = \".*\"/version = \"$NEW_VERSION\"/" "$PYPROJECT"
    rm "${PYPROJECT}.bak"
    success "Updated version in $PYPROJECT"
else
    warning "[DRY RUN] Would update version to $NEW_VERSION in $PYPROJECT"
fi

# 4. Update CHANGELOG.md
info "Updating $CHANGELOG..."
TODAY=$(date +%Y-%m-%d)

if [ "$DRY_RUN" = false ]; then
    if grep -q "## \[Unreleased\]" "$CHANGELOG"; then
        sed -i.bak "s/## \[Unreleased\]/## [Unreleased]\n\n## [$NEW_VERSION] - $TODAY/" "$CHANGELOG"
        rm "${CHANGELOG}.bak"
        success "Updated $CHANGELOG with release date"
    else
        warning "No [Unreleased] section found in $CHANGELOG, skipping"
    fi
else
    warning "[DRY RUN] Would update $CHANGELOG with release $NEW_VERSION dated $TODAY"
fi

# 5. Run tests
info "Running tests..."
if command -v pytest > /dev/null 2>&1; then
    if [ "$DRY_RUN" = false ]; then
        if ! pytest tests/ -v; then
            error "Tests failed. Fix tests before releasing."
        fi
        success "Tests passed"
    else
        warning "[DRY RUN] Would run pytest"
    fi
else
    warning "pytest not found, skipping tests"
fi

# 6. Build package
info "Building package..."
run_cmd python -m build --clean

if [ "$DRY_RUN" = false ]; then
    success "Package built successfully"
    ls -lh dist/
fi

# 7. Validate package
info "Validating package..."
if [ "$DRY_RUN" = false ]; then
    python -m twine check dist/*
    success "Package validation passed"
else
    warning "[DRY RUN] Would run twine check"
fi

# 8. Commit changes
info "Committing version bump..."
run_cmd git add "$PYPROJECT" "$CHANGELOG"
run_cmd git commit -m "chore: bump version to $NEW_VERSION"

if [ "$DRY_RUN" = false ]; then
    success "Committed version bump"
fi

# 9. Create git tag
info "Creating git tag v$NEW_VERSION..."
run_cmd git tag -a "v$NEW_VERSION" -m "Release version $NEW_VERSION"

if [ "$DRY_RUN" = false ]; then
    success "Created tag v$NEW_VERSION"
fi

# 10. Push to GitHub
info "Pushing to GitHub..."
run_cmd git push origin "$REQUIRED_BRANCH"
run_cmd git push origin "v$NEW_VERSION"

if [ "$DRY_RUN" = false ]; then
    success "Pushed commits and tags to GitHub"
fi

# 11. Publish to PyPI
if [ "$DRY_RUN" = false ]; then
    echo ""
    read -p "🚀 Ready to publish to PyPI? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        info "Publishing to PyPI..."
        python -m twine upload dist/*
        success "Published to PyPI!"
        
        echo ""
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}🎉 Release $NEW_VERSION completed successfully!${NC}"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        echo -e "${BLUE}📦 Install with: pip install squirt==$NEW_VERSION${NC}"
        echo -e "${BLUE}🔗 PyPI: https://pypi.org/project/squirt/$NEW_VERSION/${NC}"
        echo -e "${BLUE}🏷️  Tag: https://github.com/loganpowell/squirt/releases/tag/v$NEW_VERSION${NC}"
        echo ""
    else
        warning "Skipping PyPI upload"
        echo ""
        echo "To publish manually, run:"
        echo "  python -m twine upload dist/*"
    fi
else
    warning "[DRY RUN] Would prompt for PyPI upload"
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}🏃 DRY RUN COMPLETE - Review changes above${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "To perform actual release, run:"
    echo "  ./release.sh $VERSION_PART"
fi
