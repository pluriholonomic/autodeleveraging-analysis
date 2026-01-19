# Contributing to ADL Analysis

## Authors

- **Tarun Chitra** - [@pluriholonomic](https://github.com/pluriholonomic)
- **Victor Xu** - [@victators](https://github.com/victators)
- **Nagu Thogiti** - [@thogiti](https://github.com/thogiti)
- **TheBunnyAccount** - [@ConejoCapital](https://github.com/ConejoCapital)

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists in the [issue tracker](https://github.com/pluriholonomic/autodeleveraging-analysis/issues)
2. If not, create a new issue with:
   - A clear, descriptive title
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment (Python version, OS)

### Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Run the linter: `uv run ruff check src/`
5. Run the tests: `uv run pytest`
6. Ensure figures still generate: `uv run oss-adl --out ./out plots`
7. Commit your changes with a descriptive message
8. Push to your fork and submit a pull request

### Code Style

This project uses [ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check for issues
uv run ruff check src/

# Auto-fix issues
uv run ruff check --fix src/

# Format code
uv run ruff format src/
```

### Development Setup

```bash
# Clone the repository
git clone https://github.com/pluriholonomic/autodeleveraging-analysis.git
cd autodeleveraging-analysis

# Install dependencies (including dev dependencies)
uv sync --dev

# Run tests
uv run pytest

# Run linter
uv run ruff check src/
```

## Questions?

Feel free to open an issue for any questions about the codebase or methodology.
