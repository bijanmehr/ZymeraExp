# report/ — rendered results (reorganized 2026-08-09)

Rendered HTML reports + GIF galleries. Reorganized into clear categories; the interactive coverage
gallery was moved as one unit so its shared `.js` / `gifs/` still resolve.

| Folder | What | Size |
|---|---|---|
| **`planner-study/`** | The T5 **planner validation**: single-agent benchmark, multi-agent bracket, the mechanistic **why-MVProp-works proof**, distillation ablation, the planner-arm gallery, comms-layer report — plus the two **`stage0-*`** reports (classical-vs-learned, stuck-vs-covering) that motivated the planner. | 16M |
| **`coverage-results/`** | Coverage renders (progress, zero-shot versatility, wall-RF, walled planner-vs-greedy) + two sub-galleries: **`sar-zeroshot/`** (5 search-and-rescue zero-shot reports) and **`t3b/`** (4 connectivity-control reports + rollout GIFs). | 27M |
| **`belief-kb/`** | Shared-belief / KB diagnostics: KB evolution over an episode, KB-evolution with role+attention critic, and the **KB audit** (what the encoding provably carries). | 2.6M |
| **`gallery/`** | The **interactive coverage gallery** — open **`gallery/index.html`**. Data-driven pages (`pages/`), the light + combined tiers, the shared `data.js`/`nav.js`/`viewer.js`, and `gifs/`. Self-contained; don't split the pieces apart (they reference each other by relative path). | 13M |
| **`architecture/`** | Model data-sheets: policy depiction + actor/critic panels. | 40K |
| **`tools/`** | Utilities: the arena-map audit + the map maker. | 40K |
| **`_superseded/`** | **Staged for review — delete after you eyeball.** The older `mvprop-method` pass (superseded by `planner-study/`), the `_artifact` Artifact-tool copies of reports that also exist standalone, `old100gifs/`, the personality report + its 2.3M json, and a loose png. | 15M |

**Notes**
- The main entry point is now **`gallery/index.html`** (the old top-level `index.html` moved there with its assets).
- `_superseded/` is the review-dump — its contents are redundant or old; nothing current links into it.
- The lit-positioning reports that used to live here (`04_lit-positioning/`) were moved to `../docs/literature/`.
