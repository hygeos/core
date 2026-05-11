# core CHANGELOG

## v1.3.2
Release date: 2026-05-11

* Add `[tool.poetry]` packages directive in `pyproject.toml` to avoid build crash due to package name mismatch (`hygeos-core` vs `core` directory)

* Update `pixi.lock`

## v1.3.1
Release date: 2026-05-11

Hotfix release.

This version corrects the package version metadata after a mismatch with the previous release state: the last git tag was `v1.3.0`, while the version declared in `pyproject.toml` had diverged.

