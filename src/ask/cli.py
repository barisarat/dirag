"""ask CLI - typer app.

Surface (see README):
  ask "<question>"            answer over the default corpus
  ask q "<question>" [-c X]   same, with an explicit corpus
  ask index [corpus]          build/refresh a corpus index (incremental)
  ask status [corpus]         index counts
  ask config                  show resolved corpora (API keys are redacted)

A bare first argument that is not a command name is treated as a question for
the default corpus, so `ask "how are backups pruned?"` works. To ask a question
that collides with a command name, use `ask q "<question>"`.
"""

import json as json_lib
import logging
import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__, answer as answer_mod, config as config_mod, grounding, llm

cli = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Local CLI for grounded question-answering over your own PDF corpora.",
)

# Command names reserved by the CLI; anything else in first position is a question.
_COMMANDS = ("config", "q", "index", "status", "version")


def app():
    """Console entry point.

    A first argument that is not a command name (and not an option) is treated
    as a question for the `q` command, so `ask "how are backups pruned?"`
    works. Command names always win; a question that collides with one needs
    the explicit form (`ask q "config"`). The rewrite happens on argv, before
    any parsing.
    """
    argv = sys.argv[1:]
    if argv and argv[0] not in _COMMANDS and not argv[0].startswith("-"):
        sys.argv = [sys.argv[0], "q", *argv]
    return cli()


console = Console()
err_console = Console(stderr=True)

if os.getenv("ASK_DEBUG"):
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("ask").setLevel(logging.DEBUG)


def _resolve_corpus_or_exit(corpus=None, config_path=None):
    """Resolve a corpus (explicit or default), printing a clean error on failure."""
    try:
        return config_mod.resolve_corpus(corpus, config_path)
    except config_mod.ConfigError as exc:
        err_console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(code=1)


def _llm_config_or_exit(corpus, override=None):
    """Resolve the corpus's LLM target, printing a clean setup error on failure."""
    try:
        return llm.resolve_llm_config(corpus.provider, corpus.model, override=override)
    except llm.LLMSetupError as exc:
        err_console.print(f"[red]model setup:[/red] {exc}")
        raise typer.Exit(code=1)


def _citation_locator(citation):
    if citation.get("page") is not None:
        return f"page {citation['page']}"
    if citation.get("line") is not None:
        return f"line {citation['line']}"
    return ""


def _render_answer(result, json_out):
    """Render a generate_answer() result: the answer text, then numbered citations."""
    if json_out:
        console.print_json(json_lib.dumps(result))
        return

    if result.get("no_sources"):
        err_console.print("[yellow]No matching sources found.[/yellow]")
        return

    console.print(Panel(result["answer"] or "(empty answer)", title="answer", expand=True))

    citations = result.get("citations") or []
    if not citations:
        return
    table = Table(title="sources", show_lines=True)
    table.add_column("#", justify="right", style="bold")
    table.add_column("source")
    table.add_column("where")
    table.add_column("passage")
    for i, c in enumerate(citations, 1):
        table.add_row(str(i), c.get("label", ""), _citation_locator(c), c.get("passage", ""))
    console.print(table)

    if not result.get("grounded", True):
        err_console.print(
            "[yellow]note:[/yellow] the grounding check flagged content it could not tie "
            "to the sources; treat the answer with extra care."
        )
        for claim in (result.get("unsupported") or [])[:3]:
            err_console.print(f"  [yellow]- {claim}[/yellow]")


def _report_no_hits(corpus_name, counts):
    """Explain an empty retrieval: usually an unindexed or text-less corpus."""
    chunks = (counts or {}).get("chunks", 0)
    by_status = (counts or {}).get("by_status") or {}
    if chunks == 0:
        no_text = by_status.get("no_text", 0)
        if no_text:
            err_console.print(
                f"[yellow]No searchable text in '{corpus_name}'.[/yellow] {no_text} PDF(s) "
                f"had no text layer. Install ocrmypdf + tesseract and re-run "
                f"`ask index {corpus_name}`, or check `ask status {corpus_name}`."
            )
        else:
            err_console.print(
                f"[yellow]The '{corpus_name}' index is empty.[/yellow] Add PDFs to the "
                f"corpus path and run `ask index {corpus_name}` (indexing does not "
                "happen automatically)."
            )
    else:
        err_console.print(
            f"[yellow]No matching passages[/yellow] (index has {chunks} chunks). "
            "Try rephrasing, or raise --k."
        )


@cli.command()
def config(
    config_path: str = typer.Option(
        None, "--config", help="Path to corpora.toml (overrides the search order)."
    ),
):
    """Show resolved corpora and their models (API keys are redacted)."""
    try:
        corpora, default = config_mod.load_config(config_path)
    except config_mod.ConfigError as exc:
        err_console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(code=1)

    key_set = bool((os.getenv("OPENAI_API_KEY") or "").strip())

    table = Table(title="ask corpora", show_lines=False)
    table.add_column("corpus", style="bold")
    table.add_column("path")
    table.add_column("retriever")
    table.add_column("model")

    for name in sorted(corpora):
        corpus = corpora[name]
        model_ref = corpus.model_ref
        if corpus.provider == "openai" and not key_set:
            model_ref += " [red](OPENAI_API_KEY not set)[/red]"
        label = f"{name} [dim](default)[/dim]" if name == default else name
        table.add_row(label, str(corpus.path), corpus.retriever, model_ref)

    console.print(table)
    console.print(
        f"OPENAI_API_KEY: {'set (redacted)' if key_set else 'not set'}   "
        f"ollama: {llm.DEFAULT_OLLAMA_URL}"
    )


@cli.command()
def q(
    question: str = typer.Argument(..., help="Your question."),
    corpus: str = typer.Option(
        None, "--corpus", "-c", help="Corpus name (defaults to the configured default)."
    ),
    k: int = typer.Option(8, "--k", help="Number of retrieved chunks."),
    model: str = typer.Option(
        None, "--model", help="Override the corpus model as 'provider:model'."
    ),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable output."),
    config_path: str = typer.Option(None, "--config", help="Path to corpora.toml."),
):
    """Answer a question over a corpus with page-cited sources."""
    target = _resolve_corpus_or_exit(corpus, config_path)
    cfg = _llm_config_or_exit(target, override=model)

    db = _db_path(target.name)
    if not db.exists():
        err_console.print(
            f"[red]'{target.name}' is not indexed.[/red] Run `ask index {target.name}` "
            f"first (expected {db})."
        )
        raise typer.Exit(code=1)

    # Lazy imports so `config` never pulls in fastembed + sqlite-vec.
    from .library import search as lib_search
    from .library.store import Store

    store = Store(db)
    try:
        with console.status(f"searching '{target.name}'..."):
            hits = lib_search.search(store, question, k=k)

            # Bounded sufficiency re-retrieve (embed + KNN again on a reformulation).
            if hits and len(hits) < k:
                preview = "\n".join(
                    f"- {h['title']} p{h['metadata'].get('page')}: {h['text']}" for h in hits[:k]
                )
                verdict = grounding.grade_sufficiency(cfg, question, preview)
                reformulation = verdict.get("reformulation")
                if not verdict.get("sufficient", True) and reformulation:
                    extra = lib_search.search(store, reformulation, k=k)
                    seen = {h["chunkId"] for h in hits}
                    for h in extra:
                        if h["chunkId"] not in seen:
                            hits.append(h)
                            seen.add(h["chunkId"])

        counts = store.counts() if not hits else None
    finally:
        store.close()

    if not hits:
        _report_no_hits(target.name, counts)
        raise typer.Exit(code=1)

    with console.status("drafting a grounded answer..."):
        result = answer_mod.generate_answer(cfg, question, hits, max_items=k)

    _render_answer(result, json_out)


def _db_path(corpus_name):
    """Location of a corpus index (gitignored). Override the dir with ASK_DATA_DIR."""
    base = os.getenv("ASK_DATA_DIR") or "data"
    return Path(base) / f"{corpus_name}.sqlite3"


@cli.command()
def index(
    corpus: str = typer.Argument(None, help="Corpus to index (defaults to the default corpus)."),
    config_path: str = typer.Option(None, "--config", help="Path to corpora.toml."),
):
    """Build or refresh a corpus index (incremental: only changed files re-embed)."""
    target = _resolve_corpus_or_exit(corpus, config_path)
    if not target.path.exists():
        err_console.print(f"[red]corpus path does not exist:[/red] {target.path}")
        raise typer.Exit(code=1)

    # Lazy import so `config` never pulls in fastembed + sqlite-vec.
    from .library import index as index_mod
    from .library.store import Store

    db = _db_path(target.name)
    src = config_mod.find_config_path() if config_path is None else Path(config_path)
    pdfs = index_mod.find_pdfs(target.path)
    console.print(
        f"config: {src}\ncorpus path: {target.path}\nindex db: {db.resolve()}\n"
        f"found {len(pdfs)} PDF(s) under the corpus path"
    )
    if not pdfs:
        err_console.print(
            "[yellow]No PDFs found.[/yellow] Check that the corpus `path` in your config "
            f"points where your PDFs are (currently {target.path})."
        )
        raise typer.Exit(code=0)

    store = Store(db)
    try:
        # Stream progress (no hidden spinner) so a long index is observable and, if a
        # native crash kills the process, the last printed line pinpoints the step.
        report = index_mod.build_index(
            store, target.path, progress=lambda m: console.print(f"[dim]{m}[/dim]")
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean fatal instead of a raw trace
        store.close()
        err_console.print(f"[red]indexing failed:[/red] {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1)
    else:
        store.close()

    console.print(
        f"[green]indexed {target.name}[/green]: "
        f"added {report['added']}, updated {report['updated']}, "
        f"removed {report['removed']}, unchanged {report['unchanged']}, "
        f"ocr {report['ocr']}, no_text {report['skipped_no_text']}, "
        f"errors {len(report['errors'])}"
    )
    for err in report["errors"]:
        err_console.print(f"  [red]error[/red] {err['path']}: {err['error']}")
    if report["skipped_no_text"]:
        err_console.print(
            "  [yellow]note:[/yellow] some PDFs had no text layer and no OCR was "
            "available (install ocrmypdf + tesseract to index them)."
        )


@cli.command()
def status(
    corpus: str = typer.Argument(None, help="Corpus to report on (defaults to the default corpus)."),
    config_path: str = typer.Option(None, "--config", help="Path to corpora.toml."),
):
    """Show index counts for a corpus (files, chunks, last index, skipped)."""
    target = _resolve_corpus_or_exit(corpus, config_path)

    db = _db_path(target.name)
    if not db.exists():
        console.print(f"{target.name}: not indexed yet. Run `ask index {target.name}`.")
        raise typer.Exit(code=0)

    from .library.store import Store

    store = Store(db)
    try:
        c = store.counts()
    finally:
        store.close()

    by_status = ", ".join(f"{k}={v}" for k, v in sorted((c["by_status"] or {}).items())) or "-"
    console.print(f"index db: {db.resolve()}")
    console.print(
        f"[bold]{target.name}[/bold]  files={c['files']} ({by_status})  "
        f"chunks={c['chunks']}  vectors={c['vectors']}  "
        f"last_indexed={c['last_indexed'] or '-'}"
    )


@cli.command()
def version():
    """Print the ask version."""
    console.print(__version__)


if __name__ == "__main__":
    app()
