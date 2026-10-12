# Autonomous Codex: seven-day GitHub engineering cycle

This repository contains an API-powered, unattended **draft-PR** workflow: [.github/workflows/autonomous-codex.yml](../.github/workflows/autonomous-codex.yml). It is independent of ChatGPT Plus and its quota reset. It uses the official `openai/codex-action`, which requires an OpenAI **API key** and can incur additional costs. The workflow is not active until the account owner completes the steps below.

## Enable (one-time account setup)

1. Create a project-scoped API key at https://platform.openai.com/api-keys. Configure an appropriate project spending limit and monitor usage. **Never paste the key into a chat, issue or tracked file.**
2. Under https://github.com/hustlinngl/trading/settings/secrets/actions add a GitHub Actions repository secret named `OPENAI_API_KEY` with that key.
3. Under https://github.com/hustlinngl/trading/settings/variables/actions add repository variable `AUTONOMOUS_CODEX_ENABLED` with value `true`. Change it to `false` to stop subsequent scheduled runs.
4. Under repository **Settings → Actions → General → Workflow permissions**, allow Actions to create pull requests and grant the workflow its requested scoped write permissions. Keep branch protections and require review before merging.
5. Visit https://github.com/hustlinngl/trading/actions/workflows/autonomous-codex.yml and examine the first run, draft PR and current API spending. A manual **Run workflow** dispatch can test the setup.

## Schedule and proof of execution

The workflow is scheduled at 03:05, 09:05, 15:05 and 21:05 **UTC**, which is 05:05, 11:05, 17:05 and 23:05 in Italy on October 12, 2026. Scheduled API calls are gated to October 12 03:05 UTC through October 19 03:05 UTC (seven days). GitHub cron may be delayed; exact-minute starts aren't guaranteed. Each editing job has a 110-minute timeout; runs are serialized. This is approximately four work sessions per day, **not 24/7 continuous computation**, and a timeout isn't a monetary spending cap.

Actual execution is demonstrated only by a completed GitHub Actions run, its logs, the published candidate branch and the draft PR. Installing this workflow alone does not constitute a Codex run, software readiness or verified model performance. The action will fail clearly if the API secret is missing.

## What it does and what it cannot prove

Each run checks out the last persistent `agent/autonomous-codex` branch (or creates it from main), starts from trusted engineering instructions and uses Codex to choose the highest-value next improvement. It runs Python static checks, compile, tests, adversarial integrity and dashboard smoke. Failing checks prevent the automated patch from being published. A separate job publishes an approved-size diff and opens/updates one **draft PR**, never directly merging into main.

The agent is sandboxed to repository writes on Ubuntu, receives no persisted GitHub checkout credential, and cannot modify workflow files or agent instructions through the publisher's path guard. OpenAI API credentials are supplied only to the official action step. Publication happens in another job with a separate GitHub token. Code generation, backtests, and GitHub Actions checks alone cannot prove actual trading edge; model readiness requires independent economic and temporal validation. Changes created using `GITHUB_TOKEN` will not automatically trigger regular `pull_request` CI checks. Independently rerun CI and Windows packaging before merging/releasing.

## Stop or troubleshoot

Set `AUTONOMOUS_CODEX_ENABLED=false` in repository Actions variables to block new scheduled Codex jobs. Cancel an active GitHub Actions run separately. Rotate/delete the API secret to prevent new API calls. Scheduled work naturally stops after seven days; manual dispatch stays available. Check Actions run logs for failed install, missing secret, branch conflicts, test failures, pull-request permission errors, API usage exhaustion or account spending cap.

The ChatGPT scheduled reminder is **not** the worker. For a ChatGPT Plus-based alternative, the Codex desktop application's own project automation can use the subscription on a computer that remains awake with Codex running, but it is distinct from this GitHub-hosted API workflow.
