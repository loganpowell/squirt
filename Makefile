# Makefile for Sleuth development and release workflows

.PHONY: help install test lint format clean build verify release guide

help:  ## Show this help message
	@echo "Sleuth Development Commands"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick release workflow:"
	@echo "  make verify && make release-patch"

install:  ## Install package in development mode
	pip install -e ".[dev]"

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ --cov=sleuth --cov-report=html --cov-report=term

lint:  ## Run linters
	@echo "Checking imports..."
	@python -c "from sleuth import m, track, configure" && echo "✅ Imports OK"
	@echo "Checking for syntax errors..."
	@python -m py_compile sleuth/*.py && echo "✅ Syntax OK"

format:  ## Format code (requires black, isort)
	@command -v black >/dev/null 2>&1 && black sleuth/ tests/ || echo "⚠️  black not installed"
	@command -v isort >/dev/null 2>&1 && isort sleuth/ tests/ || echo "⚠️  isort not installed"

clean:  ## Clean build artifacts
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

build: clean  ## Build package
	python -m build

verify:  ## Verify package is ready for release
	./verify-release.sh

guide:  ## Show release guide
	./release-guide.sh

release-patch:  ## Release patch version (0.1.0 → 0.1.1)
	./release.sh patch

release-minor:  ## Release minor version (0.1.0 → 0.2.0)
	./release.sh minor

release-major:  ## Release major version (0.1.0 → 1.0.0)
	./release.sh major

dry-run:  ## Dry run release (patch version)
	./release.sh patch --dry-run

# Development helpers
dev-install: install  ## Alias for install

check: lint test  ## Run linters and tests

all: clean lint test build verify  ## Run full CI pipeline
