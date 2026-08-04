#!/usr/bin/env python3
"""
For each paragraph in an EPUB, insert a translated line immediately
after the original, translated line by line rather than as one
paragraph-sized blob.

Two target languages are supported per line, both optional:
- --first-lang:  what the ORIGINAL line itself is rendered as.
                 Default: none, meaning the original text is left
                 untouched (no translation call is made for it).
- --second-lang: the language of the NEW line inserted after the
                 original. Default: none, meaning no second line
                 is inserted at all (no translation call made for
                 it either).

With both left at their defaults, the script makes no Ollama calls
and just re-writes the epub as-is (still useful for --title changes
or CSS-only styling tweaks).

Optionally:
- Change the book title.
- Only process the first chapters (--debug).
- Render translations in italics.
- Render translations in a smaller font.

Usage:
    python translate_lines_epub.py input.epub output.epub

    python translate_lines_epub.py input.epub output.epub \
        --model llama3.1 \
        --title "DoubleLanguage" \
        --translation-italic \
        --translation-small

    # Keep original Spanish untouched, add an English line under it:
    python translate_lines_epub.py input.epub output.epub \
        --second-lang English

    # Replace the original with a French rendering, add German underneath:
    python translate_lines_epub.py input.epub output.epub \
        --first-lang French --second-lang German

    python translate_lines_epub.py input.epub output.epub \
        --debug

Requires:
    pip install ebooklib beautifulsoup4 requests

    Ollama running locally:
        ollama pull llama3.1
"""

import argparse
import re
import time
from dataclasses import dataclass

import requests
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, NavigableString

DEBUG_COMPILED_CHAPTERS = 15


@dataclass
class OllamaConfig:
    """Bundles all the request-tuning knobs so they don't need to be
    threaded individually through every function in the call chain."""

    timeout: int = 300
    retries: int = 1
    max_batch_lines: int = 20
    keep_alive: str = "30m"
    num_ctx: int | None = None
    num_gpu: int | None = None
    prompt_style: str = "generic"


@dataclass
class RunStats:
    """
    Accumulates Ollama's own per-request timing breakdown so you can see
    exactly where time is going: model load, prompt processing, or
    token generation. All duration fields are nanoseconds, as returned
    by Ollama's /api/generate.
    """

    calls: int = 0
    total_duration_ns: int = 0
    load_duration_ns: int = 0
    prompt_eval_duration_ns: int = 0
    prompt_eval_count: int = 0
    eval_duration_ns: int = 0
    eval_count: int = 0

    def record(self, response_json: dict) -> None:
        self.calls += 1
        self.total_duration_ns += response_json.get("total_duration", 0)
        self.load_duration_ns += response_json.get("load_duration", 0)
        self.prompt_eval_duration_ns += response_json.get("prompt_eval_duration", 0)
        self.prompt_eval_count += response_json.get("prompt_eval_count", 0)
        self.eval_duration_ns += response_json.get("eval_duration", 0)
        self.eval_count += response_json.get("eval_count", 0)

    def summary(self) -> str:
        if self.total_duration_ns == 0:
            return "No timing data yet."

        def pct(part_ns: int) -> float:
            return 100 * part_ns / self.total_duration_ns

        load_s = self.load_duration_ns / 1e9
        prompt_s = self.prompt_eval_duration_ns / 1e9
        gen_s = self.eval_duration_ns / 1e9
        gen_tok_s = self.eval_count / gen_s if gen_s > 0 else 0
        prompt_tok_s = self.prompt_eval_count / prompt_s if prompt_s > 0 else 0

        return (
            f"  [bottleneck check | {self.calls} calls]\n"
            f"    model load:     {format_duration(load_s):>8} ({pct(self.load_duration_ns):5.1f}%)\n"
            f"    prompt eval:    {format_duration(prompt_s):>8} ({pct(self.prompt_eval_duration_ns):5.1f}%) "
            f"- {prompt_tok_s:.1f} tok/s, {self.prompt_eval_count} tokens\n"
            f"    generation:     {format_duration(gen_s):>8} ({pct(self.eval_duration_ns):5.1f}%) "
            f"- {gen_tok_s:.1f} tok/s, {self.eval_count} tokens"
        )


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as e.g. '1h 23m 45s', '12m 34s', or '45s'."""

    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


# ISO 639-1 codes for the languages people are likely to pass via
# --source-lang/--first-lang/--second-lang. Falls back to the
# uppercased language name itself if not found here, which is
# harmless for prompt-building purposes.
LANGUAGE_CODES = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "dutch": "nl",
    "russian": "ru",
    "japanese": "ja",
    "chinese": "zh",
    "korean": "ko",
    "arabic": "ar",
    "polish": "pl",
    "swedish": "sv",
    "norwegian": "no",
    "danish": "da",
    "finnish": "fi",
    "greek": "el",
    "turkish": "tr",
    "czech": "cs",
    "hungarian": "hu",
    "romanian": "ro",
    "ukrainian": "uk",
    "hebrew": "he",
    "hindi": "hi",
    "vietnamese": "vi",
    "thai": "th",
    "indonesian": "id",
}


def lang_code(language: str) -> str:
    """Best-effort ISO 639-1 code lookup for a language name."""

    return LANGUAGE_CODES.get(language.strip().lower(), language.strip().upper())


def ollama_generate(
    prompt: str,
    model: str,
    ollama_url: str,
    config: OllamaConfig,
    stats: "RunStats | None" = None,
) -> str:
    """
    POST a prompt to Ollama's /api/generate, with retries on timeout.

    A slow model under CPU/GPU split load can legitimately take longer
    than expected on a big batch - rather than losing the whole run,
    retry a couple of times (with a short backoff) before giving up.

    `config.keep_alive` keeps the model resident in memory between
    calls (avoids reload overhead on every request during a long run).
    `config.num_ctx`, if set, trims the context window, which can free
    VRAM for more layers to fit on GPU.
    `config.num_gpu`, if set, controls how many layers are offloaded to
    GPU (0 = force CPU-only, useful for A/B testing whether partial GPU
    offload is actually faster than pure CPU on a given card).
    """

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": config.keep_alive,
    }
    options = {}
    if config.num_ctx:
        options["num_ctx"] = config.num_ctx
    if config.num_gpu is not None:
        options["num_gpu"] = config.num_gpu
    if options:
        payload["options"] = options

    last_error: Exception | None = None

    for attempt in range(1, config.retries + 2):  # e.g. retries=2 -> 3 total attempts
        try:
            response = requests.post(
                f"{ollama_url}/api/generate",
                json=payload,
                timeout=config.timeout,
            )
        except requests.exceptions.ReadTimeout as e:
            last_error = e
            print(
                f"  [warn] Ollama request timed out after {config.timeout}s "
                f"(attempt {attempt}/{config.retries + 1})."
            )
            if attempt <= config.retries:
                time.sleep(5)
                continue
            raise RuntimeError(
                f"Ollama request timed out after {config.timeout}s, "
                f"{config.retries + 1} attempts. Try increasing "
                "--request-timeout, lowering --max-batch-lines, "
                "or using a smaller/faster model."
            ) from e

        if response.status_code != 200:
            try:
                detail = response.json().get("error", response.text)
            except ValueError:
                detail = response.text

            raise RuntimeError(
                f"Ollama request failed ({response.status_code}): {detail}\n"
                f"Run 'ollama list' to see installed models.\n"
                f"If necessary: ollama pull {model}"
            )

        result = response.json()
        if stats is not None:
            stats.record(result)
        return result["response"].strip()

    # Unreachable, but keeps type-checkers happy.
    raise RuntimeError(str(last_error))


def translate_text(
    text: str,
    model: str,
    ollama_url: str,
    source_lang: str,
    target_lang: str,
    config: OllamaConfig,
    stats: "RunStats | None" = None,
) -> str:
    """Translate a string using a local Ollama model."""

    if not text.strip():
        return text

    if config.prompt_style == "translategemma":
        source_code = lang_code(source_lang)
        target_code = lang_code(target_lang)
        prompt = (
            f"You are a professional {source_lang} ({source_code}) to "
            f"{target_lang} ({target_code}) translator. Your goal is to "
            f"accurately convey the meaning and nuances of the original "
            f"{source_lang} text while adhering to {target_lang} grammar, "
            "vocabulary, and cultural sensitivities.\n"
            f"Produce only the {target_lang} translation, without any "
            "additional explanations or commentary. Please translate the "
            f"following {source_lang} text into {target_lang}:\n\n\n"
            f"{text}"
        )
    else:
        prompt = (
            f"Translate the following {source_lang} text to {target_lang}. "
            "Ignore broken text. "
            "Respond with ONLY the translation. "
            "Do not add notes, quotation marks, explanations, or commentary.\n\n"
            f"{text}"
        )

    return ollama_generate(prompt, model, ollama_url, config, stats)


def build_batch_prompt(
    lines: list[str],
    source_lang: str,
    target_lang: str,
    prompt_style: str = "generic",
) -> str:
    """Build a single prompt asking for all lines to be translated at once."""

    numbered = "\n".join(
        f"{i}. {line}" for i, line in enumerate(lines, start=1)
    )

    if prompt_style == "translategemma":
        # TranslateGemma's documented format is designed for a single
        # block of text, not a numbered list. We keep its persona/
        # instruction framing (which is what the model was tuned on)
        # and graft the numbering requirement onto it so batching still
        # works; if a model struggles with this, --prompt-style generic
        # or per-line calls (the automatic fallback) still work.
        source_code = lang_code(source_lang)
        target_code = lang_code(target_lang)
        return (
            f"You are a professional {source_lang} ({source_code}) to "
            f"{target_lang} ({target_code}) translator. Your goal is to "
            f"accurately convey the meaning and nuances of the original "
            f"{source_lang} text while adhering to {target_lang} grammar, "
            "vocabulary, and cultural sensitivities.\n"
            "Below is a numbered list of separate lines. Translate each "
            "line independently, keep the exact same numbering, output "
            "exactly one translated line per number, and do not merge, "
            "split, add, or remove lines. Produce only the numbered "
            "translations in the same '<number>. <translation>' format, "
            "without any additional explanations or commentary. Please "
            f"translate the following {source_lang} lines into "
            f"{target_lang}:\n\n\n"
            f"{numbered}"
        )

    return (
        f"Translate each numbered line below from {source_lang} to {target_lang}. "
        "Keep the exact same numbering, output exactly one translated line "
        "per number, and do not merge, split, add, or remove lines. "
        "Ignore broken text. Respond with ONLY the numbered translations "
        "in the same '<number>. <translation>' format, nothing else "
        "(no notes, quotation marks, or commentary).\n\n"
        f"{numbered}"
    )


def parse_batch_response(response_text: str, expected_count: int) -> list[str] | None:
    """Parse a numbered-list response. Returns None if it doesn't cleanly match."""

    parsed: dict[int, str] = {}

    for line in response_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        match = re.match(r"^(\d+)\.\s*(.*)$", line)
        if match:
            parsed[int(match.group(1))] = match.group(2).strip()

    if len(parsed) != expected_count:
        return None

    try:
        return [parsed[i] for i in range(1, expected_count + 1)]
    except KeyError:
        return None


def translate_chunk_resilient(
    chunk: list[str],
    model: str,
    ollama_url: str,
    source_lang: str,
    target_lang: str,
    config: OllamaConfig,
    stats: "RunStats | None",
) -> list[str]:
    """
    Translate one chunk of lines, adaptively recovering from failures
    instead of crashing the whole run:

    - If the request fails (e.g. all retries in ollama_generate timed
      out) and the chunk has more than one line, split it in half and
      retry each half separately. A batch that's too big for the model
      to finish in time will keep shrinking until it either succeeds
      or hits a single line.
    - If a single line still fails, it's logged and replaced with a
      visible placeholder rather than aborting the entire book.
    """

    prompt = build_batch_prompt(chunk, source_lang, target_lang, config.prompt_style)

    try:
        raw = ollama_generate(prompt, model, ollama_url, config, stats)
    except RuntimeError as e:
        if len(chunk) == 1:
            print(
                f"  [error] Giving up on this line after repeated failures, "
                f"marking as failed: {chunk[0]!r}\n    ({e})"
            )
            return [f"[TRANSLATION FAILED: {chunk[0]}]"]

        mid = len(chunk) // 2
        print(
            f"  [warn] Batch of {len(chunk)} lines failed, splitting into "
            f"{mid} + {len(chunk) - mid} and retrying..."
        )
        return translate_chunk_resilient(
            chunk[:mid], model, ollama_url, source_lang, target_lang, config, stats,
        ) + translate_chunk_resilient(
            chunk[mid:], model, ollama_url, source_lang, target_lang, config, stats,
        )

    translations = parse_batch_response(raw, len(chunk))

    if translations is None:
        if len(chunk) == 1:
            # Single line: don't re-invoke the same batch-parsing path -
            # that's what caused an infinite recursion loop before this
            # fix (a model that never wraps output in "1. ..." format
            # would fail to parse forever). Just use the raw output
            # directly, stripping a stray leading number prefix if the
            # model added one anyway.
            cleaned = re.sub(r"^\s*\d+\.\s*", "", raw).strip()
            return [cleaned]

        # The model didn't return a clean numbered list we can parse
        # reliably - fall back to one call per line, each handled
        # resiliently too, so nothing is lost.
        results = []
        for line in chunk:
            results.extend(
                translate_chunk_resilient(
                    [line], model, ollama_url, source_lang, target_lang, config, stats,
                )
            )
        return results

    return translations


def translate_lines(
    lines: list[str],
    model: str,
    ollama_url: str,
    source_lang: str,
    target_lang: str,
    cache: dict[tuple[str, str], str],
    config: OllamaConfig,
    stats: "RunStats | None" = None,
) -> list[str]:
    """
    Translate a batch of lines into `target_lang`, using `cache` to skip
    lines already translated elsewhere in the book (repeated headers,
    refrains, dialogue tags, etc.). Cache is keyed by (target_lang, line)
    so the same cache dict can safely serve multiple target languages
    at once.

    To avoid single requests ballooning into huge, slow-to-generate
    prompts (which can time out on a slow model/GPU split), lines are
    sent in chunks of at most `config.max_batch_lines` per Ollama call,
    and any chunk that still fails is adaptively split smaller rather
    than crashing the whole run (see translate_chunk_resilient).
    """

    to_translate = []
    for line in lines:
        key = (target_lang, line)
        if key not in cache and line not in to_translate:
            to_translate.append(line)

    for start in range(0, len(to_translate), config.max_batch_lines):
        chunk = to_translate[start:start + config.max_batch_lines]

        translations = translate_chunk_resilient(
            chunk, model, ollama_url, source_lang, target_lang, config, stats,
        )

        for original, translated in zip(chunk, translations):
            cache[(target_lang, original)] = translated

    return [cache[(target_lang, line)] for line in lines]


def split_into_lines(text: str) -> list[str]:
    """
    Split a paragraph's text into individual lines for separate
    translation calls.

    If the paragraph already contains hard line breaks (e.g. it came
    from <br> tags), those are used. Otherwise we fall back to
    sentence-level splitting so prose paragraphs are still broken
    into digestible chunks instead of being translated as one blob.
    """

    if "\n" in text:
        raw_lines = text.split("\n")
    else:
        raw_lines = re.split(r"(?<=[.!?])\s+", text)

    return [line.strip() for line in raw_lines if line.strip()]


def create_translation_css(
    italic: bool,
    small: bool,
) -> epub.EpubItem:
    """Create stylesheet for translated paragraphs."""

    css = [".translation {"]

    if italic:
        css.append("    font-style: italic;")

    if small:
        css.append("    font-size: 0.9em;")

    css.append("}")

    return epub.EpubItem(
        uid="translation_style",
        file_name="style/translation.css",
        media_type="text/css",
        content="\n".join(css).encode("utf-8"),
    )


def add_stylesheet_link(html_content: bytes) -> bytes:
    """Add EPUB stylesheet link to an XHTML document."""

    soup = BeautifulSoup(
        html_content,
        "html.parser",
    )

    if not soup.head:
        return html_content

    existing = soup.find(
        "link",
        href="style/translation.css",
    )

    if not existing:
        link = soup.new_tag(
            "link",
            rel="stylesheet",
            type="text/css",
            href="style/translation.css",
        )

        soup.head.append(link)

    return str(soup).encode("utf-8")


ORIGINAL_LANG_SENTINELS = {"original", "default", "none", "source"}


def is_original_sentinel(lang: str | None) -> bool:
    """True if `lang` means 'just use the original text, no translation'."""
    return lang is not None and lang.strip().lower() in ORIGINAL_LANG_SENTINELS


def add_translated_lines_to_html(
    html_content: bytes,
    model: str,
    ollama_url: str,
    source_lang: str,
    first_lang: str | None,
    second_lang: str | None,
    cache: dict[tuple[str, str], str],
    config: OllamaConfig,
    stats: "RunStats | None" = None,
) -> bytes:
    """
    For every paragraph:
      - if first_lang is set, replace the paragraph's own text with a
        translation into first_lang (otherwise leave it as the
        original text, untouched, no API call needed).
      - if second_lang is set, insert a new paragraph after it
        translated into second_lang (otherwise no second line is
        added at all, no API call needed).
    """

    soup = BeautifulSoup(
        html_content,
        "html.parser",
    )

    paragraphs = soup.find_all("p")

    for p in paragraphs:
        # Preserve <br> boundaries as newlines so poetry/dialogue lines
        # are split correctly; falls back to sentence splitting below.
        original_text = p.get_text(separator="\n", strip=True)

        if not original_text.strip():
            continue

        lines = split_into_lines(original_text)

        if first_lang and not is_original_sentinel(first_lang):
            first_lines = translate_lines(
                lines,
                model,
                ollama_url,
                source_lang,
                first_lang,
                cache,
                config,
                stats,
            )
            p.clear()
            for i, line in enumerate(first_lines):
                if i > 0:
                    p.append(soup.new_tag("br"))
                p.append(NavigableString(line))
        # else: leave the original paragraph text exactly as it is
        # (either first_lang wasn't set, or it's an "original"/"default"
        # sentinel meaning the same thing).

        if second_lang:
            if is_original_sentinel(second_lang):
                second_lines = lines  # no translation call, reuse as-is
            else:
                second_lines = translate_lines(
                    lines,
                    model,
                    ollama_url,
                    source_lang,
                    second_lang,
                    cache,
                    config,
                    stats,
                )

            translation_p = soup.new_tag("p")
            translation_p["class"] = "translation"

            translation_p.append(NavigableString("> "))
            for i, translated in enumerate(second_lines):
                if i > 0:
                    translation_p.append(soup.new_tag("br"))
                translation_p.append(NavigableString(translated))
            translation_p.append(NavigableString(" <"))

            p.insert_after(translation_p)
        # else: no second line is inserted at all.

    return str(soup).encode("utf-8")


def process_epub(
    input_path: str,
    output_path: str,
    model: str,
    ollama_url: str,
    source_lang: str,
    first_lang: str | None = None,
    second_lang: str | None = None,
    title: str | None = None,
    debug: int | None = None,
    translation_italic: bool = True,
    translation_small: bool = True,
    ollama_config: OllamaConfig | None = None,
):
    config = ollama_config or OllamaConfig()

    book = epub.read_epub(input_path)

    translation_css = create_translation_css(
        translation_italic,
        translation_small,
    )

    book.add_item(translation_css)

    doc_items = [
        item
        for item in book.get_items()
        if item.get_type() == ebooklib.ITEM_DOCUMENT
    ]

    if debug is not None:
        print(f"DEBUG MODE: processing only the first {debug} chapter{'s' if debug != 1 else ''}.")
        doc_items = doc_items[:debug]

    total = len(doc_items)

    # Shared across all chapters so repeated lines (chapter headers,
    # refrains, recurring dialogue) are only ever translated once,
    # even across different target languages (cache is keyed by
    # (target_lang, line)).
    translation_cache: dict[tuple[str, str], str] = {}

    run_start = time.time()
    chapter_durations: list[float] = []
    stats = RunStats()

    for i, item in enumerate(doc_items, start=1):
        print(
            f"Translating chapter {i}/{total}: "
            f"{item.get_name()}"
        )

        chapter_start = time.time()

        content = item.get_content()

        content = add_stylesheet_link(content)

        content = add_translated_lines_to_html(
            content,
            model,
            ollama_url,
            source_lang,
            first_lang,
            second_lang,
            translation_cache,
            config,
            stats,
        )

        item.set_content(content)

        chapter_elapsed = time.time() - chapter_start
        chapter_durations.append(chapter_elapsed)

        avg_per_chapter = sum(chapter_durations) / len(chapter_durations)
        remaining_chapters = total - i
        eta_seconds = avg_per_chapter * remaining_chapters
        total_elapsed = time.time() - run_start

        print(
            f"  chapter took {format_duration(chapter_elapsed)} | "
            f"elapsed {format_duration(total_elapsed)} | "
            f"avg {format_duration(avg_per_chapter)}/chapter | "
            f"ETA {format_duration(eta_seconds)} "
            f"({remaining_chapters} chapter{'s' if remaining_chapters != 1 else ''} left)"
        )
        print(stats.summary())

    if not title:
        existing_titles = book.get_metadata(
            "DC",
            "title",
        )
        original_title = (
            existing_titles[0][0] if existing_titles else "Untitled"
        )
        title = f"{original_title} multilang"

    book.metadata[
        "http://purl.org/dc/elements/1.1/"
    ]["title"] = []

    book.set_title(title)

    epub.write_epub(
        output_path,
        book,
    )

    print(
        f"\nDone in {format_duration(time.time() - run_start)}.\n"
        f"{stats.summary()}\n"
        f"Output written to:\n{output_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Insert a translated line after every paragraph "
            "in an EPUB using Ollama, translated line by line."
        )
    )

    parser.add_argument(
        "input",
        help="Input EPUB file",
    )

    parser.add_argument(
        "output",
        help="Output EPUB file",
    )

    parser.add_argument(
        "--model",
        default="llama3.1",
        help="Ollama model (default: llama3.1)",
    )

    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama server URL",
    )

    parser.add_argument(
        "--source-lang",
        default="Spanish",
        help="Language the original book is actually written in (default: Spanish)",
    )

    parser.add_argument(
        "--first-lang",
        default=None,
        help=(
            "Language to render the ORIGINAL line as. "
            "Default: none, meaning the original text is left "
            "untouched (no translation call made for it). "
            "Pass 'original' or 'default' explicitly for the same effect."
        ),
    )

    parser.add_argument(
        "--second-lang",
        default=None,
        help=(
            "Language for the new line inserted after the original. "
            "Default: none, meaning no second line is inserted at all "
            "(no API call made for it). Pass 'original' or 'default' to "
            "insert the original text unchanged as the second line too "
            "(no API call) - useful for checking line-splitting/alignment "
            "before spending time on real translation."
        ),
    )

    parser.add_argument(
        "--title",
        help=(
            "New title for output EPUB "
            "(default: original title + ' multilang')"
        ),
    )

    parser.add_argument(
        "--debug",
        type=int,
        nargs="?",
        const=DEBUG_COMPILED_CHAPTERS,
        default=None,
        metavar="N",
        help=(
            f"Only translate the first N chapters, e.g. '--debug 2' for "
            f"just the first 2 (default N if flag given with no value: "
            f"{DEBUG_COMPILED_CHAPTERS}). Omit entirely to process the "
            f"whole book."
        ),
    )

    parser.add_argument(
        "--translation-italic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render translated paragraphs in italics (default: on; use --no-translation-italic to disable)",
    )

    parser.add_argument(
        "--translation-small",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render translated paragraphs smaller (default: on; use --no-translation-small to disable)",
    )

    parser.add_argument(
        "--prompt-style",
        choices=["generic", "translategemma"],
        default="generic",
        help=(
            "Prompt format sent to the model. 'translategemma' uses the "
            "persona-style prompt TranslateGemma models expect "
            "(source/target language + code, two blank lines before the "
            "text). Default: 'generic', a plain instruction prompt that "
            "works fine for general-purpose models like llama3.1/qwen2.5."
        ),
    )

    parser.add_argument(
        "--request-timeout",
        type=int,
        default=300,
        help=(
            "Seconds to wait for a single Ollama response before retrying/"
            "failing (default: 300). Slower models with heavy CPU/GPU "
            "offload may need this raised."
        ),
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help=(
            "Retries per request at the SAME batch size before giving up "
            "and splitting the batch in half (default: 1, i.e. 2 total "
            "attempts). Since a batch that repeatedly times out is usually "
            "just too big rather than unlucky, low values here work well - "
            "the adaptive splitting in translate_chunk_resilient handles "
            "persistent slowness better than repeating an identical request."
        ),
    )

    parser.add_argument(
        "--max-batch-lines",
        type=int,
        default=20,
        help=(
            "Max lines translated per Ollama call (default: 20). Lower "
            "this if a slow model is timing out on long paragraphs; "
            "raise it to reduce total request count on a fast model/GPU."
        ),
    )

    parser.add_argument(
        "--keep-alive",
        default="30m",
        help=(
            "How long Ollama keeps the model loaded in memory between "
            "requests (default: '30m'). Prevents reload overhead on "
            "every call during a long run. Use '-1' to keep it loaded "
            "indefinitely, or '0' to unload immediately after each call."
        ),
    )

    parser.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help=(
            "Context window size to request from Ollama (default: model's "
            "own default, e.g. 4096). Lowering this (e.g. 2048) reduces "
            "VRAM used by the KV cache, which can let more model layers "
            "fit on GPU on VRAM-constrained cards - worth trying if "
            "'ollama ps' shows a heavy CPU/GPU split."
        ),
    )

    parser.add_argument(
        "--num-gpu",
        type=int,
        default=None,
        help=(
            "Number of model layers to offload to GPU (default: Ollama's "
            "own automatic choice). Set to 0 to force pure CPU inference - "
            "useful for A/B testing whether partial GPU offload is even "
            "faster than CPU-only on a given card, since handoff overhead "
            "between CPU and GPU layers isn't always worth it on older "
            "GPUs. Compare the 'generation tok/s' in the bottleneck check "
            "between a run with this at 0 and one without it set."
        ),
    )

    args = parser.parse_args()

    ollama_config = OllamaConfig(
        timeout=args.request_timeout,
        retries=args.retries,
        max_batch_lines=args.max_batch_lines,
        keep_alive=args.keep_alive,
        num_ctx=args.num_ctx,
        num_gpu=args.num_gpu,
        prompt_style=args.prompt_style,
    )

    process_epub(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        ollama_url=args.ollama_url,
        source_lang=args.source_lang,
        first_lang=args.first_lang,
        second_lang=args.second_lang,
        title=args.title,
        debug=args.debug,
        translation_italic=args.translation_italic,
        translation_small=args.translation_small,
        ollama_config=ollama_config,
    )


if __name__ == "__main__":
    main()
