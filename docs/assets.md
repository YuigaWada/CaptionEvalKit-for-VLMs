# Assets and Distribution

CaptionEvalKit-for-VLMs should distribute code, wrappers, manifests, and
expected-value metadata. It should not redistribute benchmark images, VLM
weights, or metric checkpoints in PyPI wheels or source distributions.

Use the asset downloader for files that have stable scriptable sources and are
not already downloaded by the metric runtime:

```bash
capevalkit download-assets --list
capevalkit download-assets
capevalkit download-assets --all
```

The default set downloads non-Hugging-Face checkpoint assets needed by common
local reproduction runs.

## Load Sources

### Models And Checkpoints

| Asset | Used by | Loaded from | Runtime behavior |
| --- | --- | --- | --- |
| OpenAI CLIP ViT-B/32 | CLIPScore, RefCLIPScore, PAC-S ViT-B/32 backbone, Polos image encoder | URL: `https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt` | Auto-downloaded by `clip.load`. |
| OpenAI CLIP ViT-L/14 | CLIPScore ViT-L/14 variants | URL: `https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt` | Auto-downloaded by `clip.load`. |
| OpenCLIP ViT-L/14 LAION2B | PAC-S OpenCLIP ViT-L/14 backbone | HF: `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` | Auto-downloaded by PACScore's OpenCLIP loader. |
| PAC-S CLIP ViT-B/32 checkpoint | `pacscore`, `refpacscore` | URL: `https://drive.usercontent.google.com/download?id=1F-0Pma-vfJPAiDzeyl-iEdSXZIO1cDae&export=download&confirm=t`; local path: `metrics/upstreams/pacscore/checkpoints/clip_ViT-B-32.pth` | Downloadable via `capevalkit download-assets`. |
| PAC-S OpenCLIP ViT-L/14 checkpoint | `pacscore-vitl`, `refpacscore-vitl`, `pacscoreavg` | Upstream README lists the same Google Drive ID as the ViT-B/32 file; local path: `metrics/upstreams/pacscore/checkpoints/openClip_ViT-L-14.pth` | Optional/manual until an authoritative source is verified. |
| PAC-S++ CLIP ViT-B/32 checkpoint | Optional PAC-S++ ViT-B/32 runs | URL: `https://ailb-web.ing.unimore.it/publicfiles/pac++/PAC++_clip_ViT-B-32.pth` | Downloadable via `capevalkit download-assets --all` or explicit asset name. |
| PAC-S++ CLIP ViT-L/14 checkpoint | `pacscorepp`, `refpacscorepp`, `pacscoreppavg` | URL: `https://ailb-web.ing.unimore.it/publicfiles/pac++/PAC++_clip_ViT-L-14.pth` | Downloadable via `capevalkit download-assets`. |
| Polos checkpoint | `polos` | URL: `https://polos-polaris.s3.ap-northeast-1.amazonaws.com/reprod.zip` | Downloadable via `capevalkit download-assets`; extracted under `.model-cache/polos/`. |
| Polos text encoder | `polos` | HF: `princeton-nlp/sup-simcse-roberta-base` | Auto-downloaded by Transformers when Polos loads its checkpoint. |
| SPICE Stanford CoreNLP 3.6.0 | `spice` | URL: `http://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip` | Auto-downloaded by pycocoevalcap SPICE on first use. |
| METEOR jar | `meteor` | Local: `metrics/upstreams/pycocoevalcap/meteor/meteor-1.5.jar` | Vendored upstream file. |
| PTBTokenizer CoreNLP jar | pycocoevalcap metrics | Local: `metrics/upstreams/pycocoevalcap/tokenizer/stanford-corenlp-3.4.1.jar` | Vendored upstream file. |
| VELA LongCLIP-L | `vela` | HF: `BeichenZhang/LongCLIP-L`, file `longclip-L.pt` | Auto-downloaded by VELA's `hf_hub_download` path. |
| VELA regressor checkpoint | `vela` regressor config | HF: `Ka2ukiMatsuda/vela`, file `vela_regressor.safetensors` | Auto-downloaded by VELA's `hf_hub_download` path. |
| VELA ranker checkpoint | Optional VELA ranker config | HF: `Ka2ukiMatsuda/vela`, file `vela_ranker.safetensors` | Auto-downloaded by VELA's `hf_hub_download` path. |
| Qwen2.5-3B-Instruct | VELA language model | HF: `Qwen/Qwen2.5-3B-Instruct` | Auto-downloaded by Transformers `from_pretrained`. |
| LLaVA v1.5 13B | FLEUR/RefFLEUR | HF: `liuhaotian/llava-v1.5-13b` | Auto-downloaded by LLaVA/Transformers `from_pretrained`. |

### Datasets

| Dataset / Benchmark | Loaded from by default | Local paths used by the loader | Notes |
| --- | --- | --- | --- |
| Composite | HF: `yuwd/Composite`, config `default`, split `test` | Rows cached under `.hf-cache/benchmarks/`; images extracted under `.hf-cache/benchmarks/composite-images/` | If a non-repo `--data-root` is passed, local `data/composite/en_test_composite_da2.csv` or `en_test_composite_da.csv` plus images are used instead. |
| Flickr8k-EX | HF: `yuwd/Flickr8k-HumanEval`, config `expert`, split `test` | Rows cached under `.hf-cache/benchmarks/`; images extracted under `.hf-cache/benchmarks/flickr8k-images/expert/` | If a non-repo `--data-root` is passed, local `data/flickr8k/flickr8k.json` plus images are used instead. |
| Flickr8k-CF | HF: `yuwd/Flickr8k-HumanEval`, config `crowdflower`, split `test` | Rows cached under `.hf-cache/benchmarks/`; images extracted under `.hf-cache/benchmarks/flickr8k-images/crowdflower/` | If a non-repo `--data-root` is passed, local `data/flickr8k/crowdflower_flickr8k.json` plus images are used instead. |
| Nebula | HF: `Ka2ukiMatsuda/Nebula` | Metadata cached under `.hf-cache/benchmarks/`; images expected under `data/nebula/images/` | Test split score correction also loads HF: `hiranohachiman/Spica`. |
| Polaris | HF: `yuwd/Polaris`, file `polaris_test.csv` | Metadata cached under `.hf-cache/benchmarks/`; images expected under `data/polaris/images/` | Only `test` split is supported by the current CSV loader. |
| LongCap-Arena | HF: `Ka2ukiMatsuda/LongCap-Arena` | Metadata cached under `.hf-cache/benchmarks/`; images extracted under `.hf-cache/benchmarks/longcaparena-images/` | If a non-repo `--data-root` is passed, local CSV/images are used instead. |

## Scriptable Assets

| Asset | Used by | Source | License status | Distribution decision |
| --- | --- | --- | --- | --- |
| `pacscore-pacs-vitb` | PAC-S CLIP ViT-B/32 | PACScore Google Drive file | Checkpoint-specific license is not declared upstream | Scriptable download into the upstream checkpoint directory; do not bundle. |
| `pacscore-pacs-openclip-vitl` | PAC-S OpenCLIP ViT-L/14 | PACScore Google Drive file listed by upstream | Checkpoint-specific license is not declared upstream | Optional explicit asset only; upstream link is ambiguous. |
| `pacscore-pacspp-vitb` | PAC-S++ CLIP ViT-B/32 | PAC-S++ public URL | Checkpoint-specific license is not declared upstream | Scriptable download only; do not bundle. |
| `pacscore-pacspp-vitl` | PAC-S++ CLIP ViT-L/14 | PAC-S++ public URL | Checkpoint-specific license is not declared upstream | Scriptable download only; do not bundle. |
| `polos-reprod` | Polos | Polos S3 archive | Upstream code is BSD-3-Clause-Clear; checkpoint is provided by the upstream project | Scriptable download into `.model-cache/polos`; do not bundle. |

## Assets Not Automated

| Asset | Reason | Recommended handling |
| --- | --- | --- |
| PAC-S OpenCLIP ViT-L/14 checkpoint | The local upstream README lists the same Google Drive ID as the CLIP ViT-B/32 file, so the direct OpenCLIP link is ambiguous without an authoritative checksum. | Keep optional/manual instructions until the upstream source or checksum is verified. |
| VELA LongCLIP-L | The VELA loader calls Hugging Face `hf_hub_download` and copies the checkpoint into the expected local path when the model is first loaded. | Do not include in `download-assets`; rely on the runtime loader or standard Hugging Face cache prewarming outside this tool. |
| VELA regressor/ranker checkpoints | The VELA loader calls Hugging Face `hf_hub_download` for `Ka2ukiMatsuda/vela` checkpoints. | Do not include in `download-assets`; rely on the runtime loader. |
| Qwen2.5-3B-Instruct | VELA passes `Qwen/Qwen2.5-3B-Instruct` to Transformers `from_pretrained`, which downloads from Hugging Face on first load. The model card declares `qwen-research`. | Do not include in `download-assets`; users must follow the upstream license and Hugging Face access flow. |
| LLaVA v1.5 13B | FLEUR passes `liuhaotian/llava-v1.5-13b` to LLaVA/Transformers loaders, which download from Hugging Face on first load. The model card states Llama 2 Community License. | Do not include in `download-assets`; users must follow the upstream license and Hugging Face access flow. |
| Composite/Flickr8k/COCO images | Image datasets have their own distribution terms and are not small package assets. | Composite and Flickr8k are loaded from the project HF datasets by default; users who need a local override can place original-provider files under `data/` and pass a non-repo `--data-root`. |
| Nebula images | The Hugging Face dataset card does not expose a clear license in the loader-visible metadata. | Allow runtime/user-provided dataset access; do not mirror or bundle. |
| Polaris images | The Hugging Face dataset card declares a permissive license, but the dataset is large and image provenance can still matter for downstream users. | Use Hugging Face runtime caching; do not bundle in PyPI. |
| LongCap-Arena images | The Hugging Face dataset card declares BSD-3-Clause-Clear, while VELA's upstream notes refer users to SA-1B license acceptance for original images. | Use Hugging Face runtime caching or user-provided local data; do not bundle. |

## PyPI Policy

The PyPI package should provide the `capevalkit` command and the Python package
only. It should not run post-install downloads, because that makes installs
large, slow, network-dependent, and license-ambiguous. Download commands should
remain explicit user actions so users can review upstream licenses and cache
large files in a controlled location.
