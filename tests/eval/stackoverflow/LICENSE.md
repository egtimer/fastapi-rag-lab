# Stack Overflow derived data

The files in this directory (`dev.json`, `test.json`, `questions.jsonl`) contain
question titles and metadata from Stack Overflow (https://stackoverflow.com),
obtained through the Stack Exchange API. User contributions on Stack Overflow
are licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
This derived data is distributed under the same license.

- Source: questions tagged `fastapi` on Stack Overflow, created after
  2018-05-02 (when CC BY-SA 4.0 took effect).
- Attribution: every item in `questions.jsonl` carries the question `link` and
  the author (`author_display_name`, `author_link`). Retain these when reusing.
- Changes made: only question titles are kept (no question or answer bodies);
  titles are HTML-unescaped. Labels (FastAPI docs pages) were derived from
  links found in answers, which are not redistributed.
