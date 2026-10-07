# Alpha UI Blueprint — Anime Signal Cockpit

The Alpha presentation layer is intentionally decoupled from the trading/research engine.

## Visual direction

**Name:** Sakura Tactical / Anime Intelligence Cockpit

- Dark near-black base with restrained pink/magenta neon accents.
- Soft glass panels, thin luminous borders and subtle depth rather than heavy gradients everywhere.
- Anime-inspired operator character as a visual companion, not as a source of trading decisions.
- Decorative tactical motifs can include stylized weapons, blades, charms, warning seals and HUD ornaments. They remain purely cosmetic and never represent executable actions.
- Pink is the identity accent; green/red remain reserved for market direction and outcome semantics.
- The interface should feel premium, calm and fast rather than like a game HUD overloaded with effects.

## Motion contract

Target **60 FPS** on a normal desktop.

Use GPU-friendly `transform`, `opacity`, `filter` and compositor layers for continuous animation. Avoid animating layout properties such as `top`, `left`, `width`, `height`, `box-shadow` at high frequency.

Three animation classes:

1. **Ambient** — slow breathing glow, character idle motion, grid drift.
2. **State** — signal cards, regime transitions, data freshness, outcome resolution.
3. **Interaction** — hover lift, focus ring, cursor response, panel expansion.

All motion must have a `prefers-reduced-motion` fallback.

## Cursor

Use a small custom anime-girl cursor asset with a transparent background.

- Primary cursor: expressive anime face/upper-body silhouette with pink hair.
- Hotspot kept near the hand/face so click accuracy remains normal.
- 32–48 px visual footprint; never obscure chart data.
- Fallback: native pointer if the custom asset fails.
- On interactive controls, use a subtle glow/ring rather than changing the cursor into a pointer-sized animation.

The final cursor art should be a lightweight optimized SVG/WebP/PNG asset, not a large background image.

## Information hierarchy

The Alpha home screen should answer these questions in under two seconds:

**What is happening?**
- Selected asset and live price.
- Current model signal.
- Signal strength / robust edge.
- Data freshness.
- Whether the deployment bundle is compatible.

**Why?**
- Probability.
- Robust expected-return interval.
- Meta-policy confidence.
- Model disagreement.
- Analog-memory agreement and support.
- Current regime and persistence.
- Duration specialist verdict.

**What happened before?**
- Signal markers directly on the chart.
- Prequential journal.
- WIN / LOSS / TIMEOUT / AMBIGUOUS outcome history.
- Small rolling reliability summaries.

## Layout

### 1. Command header

Persistent status bar with:
- terminal readiness;
- exchange/timeframe;
- live connection pulse;
- selected asset;
- last model scan;
- last realtime quote.

### 2. Hero market canvas

Large cinematic chart with:
- candles;
- close path;
- realtime price line;
- signal markers;
- regime shading;
- hover crosshair;
- compact tooltip;
- animated transition when changing asset/range.

The chart itself must remain readable first. Decorative anime/HUD elements should live in the perimeter, never over important candles.

### 3. Signal operator card

Selected signal receives one dominant card:
- LONG / SHORT / WAIT;
- confidence;
- robust edge band;
- score;
- decision timestamp;
- concise machine-readable reason chips.

The card can expand into a detail drawer without navigating away.

### 4. Intelligence rail

Stacked modules for:
- model consensus;
- meta-policy;
- analog memory;
- regime;
- duration specialist;
- provenance/evidence.

Modules should appear progressively so the first frame stays fast.

### 5. Signal timeline

Horizontal timeline connecting:
prediction → observation → maturity → outcome.

This is the main Alpha storytelling element: the user can see what the model said and what the market subsequently did.

### 6. Research/evidence drawer

Optional deep inspection:
- training rows and period;
- holdout metrics;
- dataset fingerprint;
- deployment semantics fingerprint;
- artifact fingerprint;
- compatibility reason;
- data age and quality.

Nothing here should be editable from the dashboard.

## Microinteractions

- Signal changes: 150–220 ms crossfade + scale transition.
- Asset changes: chart crossfade, then smooth data interpolation.
- Fresh tick: realtime line/value updates with no layout shift.
- New signal: subtle radial pulse around the signal marker.
- Outcome resolution: journal row animates from OPEN to final outcome.
- WAIT: calm amber pulse, never an aggressive alarm.
- Compatibility failure: explicit WAIT state with reason, not a hidden red error.

## Performance rules

- Keep the dashboard single-request bootstrap friendly.
- Reuse cached state and history where possible.
- Batch realtime ticker requests when the exchange adapter supports it.
- Never run high-frequency model inference in the browser.
- Keep continuous animation isolated from data rendering.
- Limit particle/decorative effects and disable them on low-power/reduced-motion settings.
- Do not let chart animation block quote polling or model refresh.

## Data/API contract

Keep the existing read-only boundary stable:

- `/api/state` — model state, signals, journal and provenance.
- `/api/history` — closed-candle OHLC history.
- `/api/quote` — display-only realtime ticker.
- `/api/health` — terminal health.

No order route is part of the Alpha UI contract.

## Anime art system

Use a small reusable asset family rather than placing unrelated characters everywhere:

- **Operator:** main anime girl, used sparingly in hero/empty states.
- **Scout:** small secondary pose for market/radar modules.
- **Guardian:** defensive motif for provenance/safety/evidence.
- **Weapon ornaments:** stylized blades/firearms/mecha silhouettes as non-interactive HUD decoration only.
- **Signal glyphs:** custom LONG, SHORT and WAIT emblems.

Art direction should stay coherent: same palette, line weight, eye/highlight language and lighting. The characters must never compete with price/decision information.

## Accessibility and resilience

- Maintain keyboard navigation.
- Maintain readable contrast for all semantic colors.
- Support reduced motion.
- Custom cursor always has a native fallback.
- Dashboard remains usable if realtime quotes fail.
- Dashboard remains usable if decorative assets fail.
- Core data and signal semantics must remain understandable without animation.

## Alpha success criterion

The user should be able to open the app and instinctively understand:

**market → signal → reasons → evidence → historical outcome**

without visiting a second page, while the system still feels like a polished anime intelligence console rather than a generic trading template.
