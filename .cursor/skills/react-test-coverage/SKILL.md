---
name: react-test-coverage
description: Audits React component test coverage, produces a structured coverage report, and optionally publishes it to Confluence. Use when the user asks for React test coverage, frontend testing overview, component test gaps, or to upload a coverage report to Confluence.
---

# React Component Test Coverage

## Scope

Audit React/TSX UI test coverage in the current repo, summarize gaps, and optionally publish the report to Confluence.

**This repo:** frontend at `retail_forecasting_optimization/frontend/`; closest backend coverage in `retail_forecasting_optimization/tests/test_dashboard_api.py`.

## Workflow checklist

```
Coverage audit:
- [ ] Inventory React components (TSX/JSX under frontend/)
- [ ] Search for test files (*.test.*, *.spec.*)
- [ ] Inspect package.json for test runner / coverage tooling
- [ ] Check for E2E tools (Playwright, Cypress)
- [ ] Identify indirect coverage (API tests the UI calls)
- [ ] Write structured report (see template below)
- [ ] Upload to Confluence if requested
```

## Step 1: Inventory components

From repo root:

```bash
find retail_forecasting_optimization/frontend/src -name '*.tsx' -o -name '*.jsx' | sort
wc -l retail_forecasting_optimization/frontend/src/**/*.{tsx,ts} 2>/dev/null | sort -n
```

Classify files:
- **Pages** — `pages/*`
- **Components** — `components/*`
- **App shell** — `App.tsx`, `main.tsx`
- **Supporting** — `api/`, `types.ts` (not components, but note if untested)

## Step 2: Check test infrastructure

Read `frontend/package.json`:
- `scripts`: is there a `test` or `coverage` script?
- `devDependencies`: Vitest, Jest, `@testing-library/*`, Playwright, Cypress?

Search repo for frontend tests:

```bash
find retail_forecasting_optimization/frontend -name '*.test.*' -o -name '*.spec.*' 2>/dev/null
```

**Zero test files + no test runner = 0% React component coverage.**

## Step 3: Find indirect coverage

Search Python/other backend tests for endpoints the UI consumes. In this repo:

```bash
grep -l 'dashboard_api\|/api/' retail_forecasting_optimization/tests/*.py
```

Read matching tests and list which `/api/*` routes are covered. This is **API contract coverage**, not UI coverage — call that out explicitly.

## Step 4: Write the report

Use this template:

```markdown
# React Component Test Coverage

**Report date:** [today from user_info]
**Repository:** [repo name / frontend path]

## Summary

[One sentence: is there React test coverage? Effective % if calculable.]

## Frontend inventory

| File | Lines | Role |
|------|------:|------|
| ... | ... | ... |

## Testing infrastructure

[package.json scripts, devDependencies, test file count]

## What is tested (non-React)

[Backend/API tests that feed the UI — endpoints covered, what's still untested in UI]

## Coverage summary

| Layer | Covered? | Notes |
|-------|----------|-------|
| React components | ... | ... |
| API client | ... | ... |
| Dashboard API (backend) | ... | ... |
| E2E / visual | ... | ... |

## Recommended priorities

1. [Highest-value, lowest-effort targets first]
2. ...
```

Prioritize recommendations by effort vs value:
1. Shared UI states (`LoadingState`, `ErrorState`, empty states)
2. Routing / layout
3. Page data-fetch flows (mock API)
4. Complex pages (charts, filters, view modes)

Default test stack suggestion for Vite projects: **Vitest + React Testing Library**; optional **Playwright** for smoke E2E.

## Step 5: Publish to Confluence (when requested)

Use Atlassian MCP (`plugin-atlassian-atlassian`):

1. `getAccessibleAtlassianResources` — get `cloudId`
2. `getConfluenceSpaces` — pick target space
   - Default: user's personal space (match `spaceOwnerId` to current user from `atlassianUserInfo`)
   - Use a named team space only if the user specifies one
3. `createConfluencePage` with:
   - `cloudId`, `spaceId`, `title`, `body`, `contentFormat: "markdown"`
   - Title pattern: `React Component Test Coverage — [Project Name]`

Return the Confluence page URL to the user.

## Repo-specific notes

| Path | Purpose |
|------|---------|
| `retail_forecasting_optimization/frontend/` | React dashboard (~8 TSX files) |
| `retail_forecasting_optimization/tests/test_dashboard_api.py` | FastAPI endpoint tests |
| `retail_forecasting_optimization/.venv/bin/python -m pytest -q` | Run existing (Python-only) tests |

No frontend linter or test runner is configured in this repo today.

## Additional resources

- Confluence upload details and HTML formatting notes: [reference.md](reference.md)
