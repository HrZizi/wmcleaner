# Contributing to wmcleaner

Thanks for your interest. wmcleaner is a small, focused tool, and the most
useful contributions are targeted fixes, better tests with synthetic fixtures,
and new source adapters.

## Ground rules

1. **Keep changes scoped.** Prefer targeted edits over broad rewrites, and do
   not modify unrelated code.
2. **Never add third-party content.** Do not commit comic/manhua/manga pages,
   scraped chapter content, ground-truth corpora, comparison archives
   (`.rar`/`.zip`), databases, model weights, or local `.env` files. Use
   synthetic or generated fixtures instead.
3. **Preserve existing behaviour** unless your change explicitly requires
   altering it.
4. **Respect the intent of the project.** wmcleaner is for cleaning watermarks
   from images the user is authorized to modify.

## Development setup

Python 3.10–3.11 is required.

```bash
# Windows
setup_core.cmd

# Linux
./setup_linux.sh core
```

Install the package in editable mode if you prefer a manual setup:

```bash
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

## Running tests

```bash
python -m pytest tests -q
```

Tests must run without any real comic-page fixtures. If a change needs new
coverage, add a **synthetic** fixture generated in the test or by a script in
`docs/`.

## Adding a source adapter

Implement the `SourceScraper` interface in `wmcleaner/scrapers/base.py`, then
register it in `wmcleaner/scrapers/registry.py`. Keep the downloader sequential
and rate-limited; do not remove the request delay. Do not add adapters whose
purpose is bulk unauthorized acquisition.

## Regenerating documentation images

```bash
python docs/generate_examples.py
```

This redraws the synthetic example artwork and rewrites the images under
`docs/images/`. The generator uses only original vector artwork plus the
project's own detector template.

## Pull requests

- Run `python -m pytest tests -q` before opening a PR and note the result.
- Describe what changed, why, and which tests cover it.
- Call out any change to detection thresholds, masks, or backend routing, as
  these affect real output.

## License

By contributing you agree that your contributions are licensed under the MIT
License (see [LICENSE](LICENSE)).
