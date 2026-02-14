# Agent Guidelines

These guidelines are intended for AI agents working on this codebase. Please follow them strictly.

## 1. Environment Management (uv)

This project uses `uv` for dependency and environment management. When running commands or scripts:

- **Activate the environment**: `source .venv/bin/activate`
- **Or use `uv run`**: Prepend commands with `uv run` (e.g., `uv run python script.py`, `uv run pytest`)

## 2. Test-Driven Development (TDD)

Adopt a TDD workflow whenever possible and sensible:

1.  **Write Tests First**: Where appropriate, create a new test file or add test cases to an existing file that cover the desired functionality.
2.  **Verify with User**: Present the proposed tests to the user and ask: "Do these tests look good?"
3.  **Implement Code**: Write the implementation code once the tests are ready or the approach is clear.
4.  **Green State**: Iterate on the code until all tests pass.

## 3. Documentation Maintenance

After completing code changes, always verify the documentation:

- **Check README.md**: unique meaningful changes often affect the README.
- **Update Accordingly**: If your changes modify usage, setup, or features described in the README, update it immediately to reflect the current state of the project.

## 4. Code Comments

- **Self-Documenting Code**: Avoid descriptive comments that merely repeat what the code does (e.g., `# 1. Load data`).
- **Explain "Why", Not "What"**: Only leave comments to explain non-obvious logic, complex workarounds, or decisions that might be confusing to a future reader.
