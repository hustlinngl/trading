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

## Microinteractions and fancy effects

The Alpha should feel alive. Fancy effects are encouraged as long as they are layered on top of a stable information hierarchy.

- Signal changes: 150–220 ms crossfade + scale transition, followed by a short radial glow pulse.
- Asset changes: chart crossfade, smooth data interpolation and a brief perimeter sweep.
- Fresh tick: realtime price/value updates with a tiny neon shimmer, never a layout shift.
- New signal: marker bloom + soft expanding ring + a short HUD sweep; keep the effect under ~700 ms.
- Outcome resolution: journal row transitions from OPEN to WIN/LOSS/TIMEOUT/AMBIGUOUS with a small glow and directional micro-particle burst.
- WAIT: calm amber breathing glow, never an aggressive alarm.
- Compatibility failure: explicit WAIT state with an amber/red diagnostic bloom and reason, never a hidden error.
- Panel hover: 2–4 px visual lift, border glow and soft ambient halo.
- Menu buttons: animated underline/edge trace on hover, stronger glow on active state.
- Click feedback: every clickable control gets a tactile response: 80–140 ms scale-down, center-out ripple, temporary neon bloom and quick return to rest.
- Menu opening: 160–240 ms fade/translate + blur reduction; do not use large elastic/bouncy motion.
- Menu switching: old panel fades/slides 20–40 px while the new panel enters with a short glow sweep.
- Buttons can use animated gradient borders and “energy” strokes around their perimeter, but motion must remain subtle enough not to distract from the chart.
- Long/Short controls use semantic glow colors; WAIT uses amber. Pink/magenta is the interface identity and should remain visible in neutral states.
- Focus/keyboard interaction receives the same visual polish as mouse interaction.
- Cursor proximity may create a very small local glow around interactive elements; never make this interfere with chart crosshairs or text selection.

## Global atmosphere

Use several slow, layered effects to give the cockpit a premium anime/HUD feel:

- faint moving star/petal dust in the background;
- soft radial Sakura glows behind important modules;
- thin animated HUD lines around the hero chart;
- occasional scanline/sweep accents, used sparingly;
- pulsing status LEDs for connection/data freshness;
- low-amplitude character idle animation;
- occasional tiny spark/cherry-blossom particles near major state changes.

These effects should be event-driven or slow-moving, never a constant full-screen particle storm.

## Glowing design language

Glow is part of the visual identity, not an afterthought.

- Base UI: subtle pink ambient halo around active panels.
- Hover: border + local shadow/glow.
- Active: stronger border illumination + inner bloom.
- Click: short-lived bright pulse.
- LONG: green signal glow.
- SHORT: red signal glow.
- WAIT: amber glow.
- New data: pink/blue accent shimmer.
- Errors: red diagnostic pulse.
- Provenance/guardian elements: cool cyan/purple secondary glow.

Avoid making every surface permanently luminous. Contrast between glow and calm areas is what makes the important events feel powerful.

## Performance rules

- Keep the dashboard single-request bootstrap friendly.
- Reuse cached state and history where possible.
- Batch realtime ticker requests when the exchange adapter supports it.
- Never run high-frequency model inference in the browser.
- Keep continuous animation isolated from data rendering.
- Prefer compositor-friendly animation (transform, opacity, limited filter).
- Use CSS variables for theme/glow intensity rather than rewriting large inline style trees.
- Use event-driven particles rather than continuously spawning DOM nodes.
- Limit particle/decorative effects and disable or simplify them on low-power/reduced-motion settings.
- Do not let chart animation block quote polling or model refresh.
- Fancy effects must degrade gracefully on weaker GPUs and high-DPI displays.

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
