# Claude Code Notes

## Reference Projects

### Task Management with rq
- Use rq for all task management. rq is in `../rq`
- See `~/context/projects/rq.md` for full rq philosophy, CLI reference, and task granularity guidance
- **Quick start**: `cd ../rq && go build -o rq ./cmd/rq` then `../rq/rq context` for protocol docs

### Experiment Management (piensa)
- See `~/context/projects/piensa.md` for patterns

### Distributed Compute (tsdev-genai.poc)
- See `~/context/projects/distributed-compute.md` for patterns

## Python / uv

### Dev workflow
- `uv sync` -- single setup command (installs all deps including dev group)
- `uv run memz serve` -- run the CLI
- `uv run pytest` -- run tests
- `uv add <pkg>` -- add a runtime dependency
- `uv add --group dev <pkg>` -- add a dev dependency

### Packaging conventions
- Build backend: hatchling (auto-discovers src/ layout)
- Source layout: `src/memz/`
- PEP 561 marker: `src/memz/py.typed`
- Lock file (`uv.lock`) is committed for reproducible installs
- Dev deps live in `[dependency-groups]` (PEP 735), not `[project.optional-dependencies]`
- `.python-version` pins the interpreter version for uv
