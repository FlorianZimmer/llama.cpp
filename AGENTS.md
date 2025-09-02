# AGENTS

This file provides guidance for contributors and automation working in this repository.

## Testing
- Build the project with `cmake -B build`.
- Compile with `cmake --build build`.
- Run the test suite with `ctest --test-dir build`.
- If only Python code is touched, run `pytest` instead of the CMake steps.

## Code Style
- Follow the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md).
- Format C and C++ sources with `clang-format` (version 15+).
- Use four spaces for indentation and keep lines free of trailing whitespace.

## Pull Requests
- Keep changes focused and well described.
- Document any new command-line options or APIs.
- Include test results in the PR description.
