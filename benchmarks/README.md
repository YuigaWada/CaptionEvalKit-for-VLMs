# Benchmarks

CaptionEvalKit treats metric evaluation as a benchmark run plus a reported-value check.

Target benchmark names:

- `composite`
- `flickr8k-ex`
- `flickr8k-cf`
- `nebula`
- `polaris`
- `longcaparena-testa-desc`
- `longcaparena-testa-rel`
- `longcaparena-testa-flu`
- `longcaparena-testb-desc`
- `longcaparena-testb-rel`
- `longcaparena-testb-flu`

The implemented runner contract is:

```text
capevalkit benchmark --metric <metric> --benchmark <benchmark> --data-root /path/to/data --output /outputs/<metric>/<benchmark>.json -- <metric-specific args>
capevalkit verify --results /outputs/<metric>/<benchmark>.json --expected benchmarks/expected/<metric>/<benchmark>.json --round-decimals 1
```

The expected JSON files contain numeric values reported by the corresponding paper or official metric repository.
`verify` compares every numeric value in the expected file against the result file with a configurable tolerance.
The paper table values are rounded to one decimal place, so prefer `--round-decimals 1`.

Current concrete mapping:

- Generic benchmark mode converts `composite`, `flickr8k-ex`, `flickr8k-cf`, `nebula`, `polaris`, and LongCap-Arena perspective splits into JSONL, runs the metric scoring command, then computes Kendall tau-b/tau-c.
- `nebula` is loaded from Hugging Face `Ka2ukiMatsuda/Nebula` by default. For the test split, scores are corrected from matching `imgid + mt` rows in Hugging Face `hiranohachiman/Spica`, matching the arXiv:2512.21582 note that Nebula annotation errors were corrected.
- `polaris` is loaded from Hugging Face `yuwd/Polaris` by default.
- `longcaparena-testa-{desc,rel,flu}` maps to Hugging Face `Ka2ukiMatsuda/LongCap-Arena` splits `dci_val.{desc,rel,flu}`.
- `longcaparena-testb-{desc,rel,flu}` maps to Hugging Face `Ka2ukiMatsuda/LongCap-Arena` splits `dci_test.{desc,rel,flu}`.
- `bleu`, `rouge`, `meteor`, `cider`, and `spice` do not read image files; they can be validated as soon as the benchmark text files are available.
- `clipscore`, `clipscore-vitl`, `clipscoreavg`, `refclipscore*`, `pacscore*`, `refpacscore*`, `fleur`, `reffleur`, `polos`, and `vela` need image files for full benchmark validation.
- `vela` uses the official `Ka2ukiMatsuda/VELA` code and the public VELA checkpoint from Hugging Face.
- `fleur` and `reffleur` use the official `Yebin46/FLEUR` code path with the `metrics/upstreams/fleur/LLaVA` submodule.
- `polos --native --benchmark flickr8k-cf` maps to the existing `metrics/upstreams/polos/validate/validate_cvpr.py --flickr`.
- `polos --native --benchmark flickr8k-ex` maps to the existing `metrics/upstreams/polos/validate/validate_cvpr.py --flickr`.
- `polos --native --benchmark polaris` maps to the existing `metrics/upstreams/polos/validate/validate_cvpr.py --coef`.

The expected values under `benchmarks/expected/{rouge,meteor,cider,spice,clipscore,pacscore,polos}/` are copied from arXiv:2512.21582 Table 1.
The BLEU values are copied from the same table except `flickr8k-cf`, where the official PACScore README correlation output gives `16.9` while arXiv:2512.21582 lists `16.4`.
The BLEU `polaris` value is copied from Polaris benchmark tables that report BLEU-4 Kendall tau-c.
The VELA LongCap-Arena values are copied from Matsuda et al. 2025 Table 1 and Appendix D Table 4. The paper reports VELA as mean±std over five runs; CaptionEvalKit evaluates the single public checkpoint, so VELA LongCap-Arena tasks use a wider reproduce tolerance while still printing `actual/paper` values.
The RefCLIPScore and RefPAC-S Polaris values are copied from the Polos paper table, while the Nebula values are copied from the DENEB paper table.
The FLEUR and RefFLEUR short-caption values are copied from the FLEUR paper.
Only VELA LongCap-Arena expected values are shipped because they are the LongCap-Arena targets in the default `all_reproduce` suite. Other LongCap-Arena baselines from the VELA paper and GPT/G-VEval style API metrics are intentionally not included in the local default target set.
