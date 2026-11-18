# Haiku with and without forced structured output, four runs each

Same 50 dev answers (`2026-10-30-judge-items-so-dev-50`), same prompt (v1), Haiku 5.5 at
effort `low`. "Plain" asks for a JSON object in text; "structured" forces it through the output
schema (`--structured`). Each setting has four runs: the original one (2026-10-30 plain,
2026-11-10 structured) and three new ones on 2026-11-18. The settings were chosen after the
labels were known, so this is a stability check of an exploration, not a held-out result.
Cost of the six new runs: $0.16.

`agreement.json` has every pairwise comparison. The figures below are recomputed on the 45
items that every run parsed and that I labelled, so both settings are scored on the same
answers.

| Setting | Unparsed replies per run | Kappa with me, same 45 items | False pass, same 45 items |
|---|---|---|---|
| Plain | 3, 2, 2, 3 | 0.72, 0.59, 0.72, 0.62 (mean 0.66) | 14%, 21%, 11%, 14% (mean 15%) |
| Structured | 0, 0, 0, 0 | 0.56, 0.47, 0.65, 0.59 (mean 0.57) | 29%, 32%, 25%, 21% (mean 27%) |

- **The effect repeats.** With the schema forced, Haiku accepts more answers I labelled
  `unsupported` in every run, and its kappa is lower on average. The ranges overlap at the
  edges (0.59 to 0.72 against 0.47 to 0.65), so four runs show the direction, not its size.
- **The unparsed items do not explain it.** The five answers the plain judge failed to parse in
  some run (j-01, j-06, j-10, j-19, j-46) are all ones I labelled `unsupported`, and the
  structured judge called all five `unsupported` in all four runs. The earlier note that "part
  of the drop is which items get counted" was wrong: on identical items the gap remains.
- **Consequence.** The structured setting fixes parsing and makes the judge more lenient, so it
  is not a free fix. The reported judge (Sonnet, plain) is unaffected; `judge_config` records
  `structured_output` and the schema hash, so a switch would require re-judging the labelled
  sets.
