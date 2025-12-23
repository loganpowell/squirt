# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial public release
- Core `@track` decorator for instrumentation
- Auto-injection of runtime_ms, memory_mb, cpu_percent metrics
- Built-in metrics: runtime_ms, memory_mb, error_free, structure_valid, expected_match
- MetricBuilder with auto-derivation of aggregation types from system metrics
- Two-tier metrics system (component-level → system-level)
- Pytest plugin with zero-config integration
- CLI tool for report generation
- Historical tracking and sparkline trends
- GitHub Actions integration
- Contrib plugins: vector, llm, chunk, data
- Comprehensive documentation and examples

### Changed

- None (initial release)

### Deprecated

- None

### Removed

- None

### Fixed

- None

### Security

- None

## [0.1.0] - 2025-12-23

### Added

- Initial beta release
- Core metrics framework
- Plugin system
- Report generation
- Pytest integration
- CLI interface

[Unreleased]: https://github.com/loganpowell/squirt/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/loganpowell/squirt/releases/tag/v0.1.0
