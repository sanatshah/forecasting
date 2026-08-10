# React Test Coverage — Reference

## Confluence upload

### MCP tools (server: `plugin-atlassian-atlassian`)

| Tool | When |
|------|------|
| `atlassianUserInfo` | Identify current user for personal-space default |
| `getAccessibleAtlassianResources` | Resolve `cloudId` |
| `getConfluenceSpaces` | List spaces; filter `status: "current"` |
| `createConfluencePage` | Publish report |

### createConfluencePage parameters

```json
{
  "cloudId": "<from getAccessibleAtlassianResources>",
  "spaceId": "<numeric id or space key>",
  "title": "React Component Test Coverage — Forecasting Dashboard",
  "body": "<markdown report>",
  "contentFormat": "markdown",
  "status": "current"
}
```

Page is created as a child of the space homepage by default.

### Space selection

- **Personal space** — use when user says "upload to Confluence" with no space named. Match `spaceOwnerId` to the account from `atlassianUserInfo`.
- **Team space** — only when user names it (e.g. `GrafanaDem`, `MISC`).

## Signals for zero frontend coverage

- No `*.test.tsx`, `*.spec.tsx`, `*.test.jsx`, `*.spec.jsx` under frontend
- `package.json` lacks `vitest`, `jest`, `@testing-library/react`, `playwright`, `cypress`
- No `test` or `coverage` npm script

## Indirect coverage mapping (this repo)

Frontend `api/client.ts` calls these routes; `test_dashboard_api.py` covers:

| Endpoint | Tested in pytest? |
|----------|-------------------|
| `/api/health` | Yes |
| `/api/summary` | Yes |
| `/api/action-breakdown` | Yes |
| `/api/recommendations` | Yes |
| `/api/metrics/department` | Yes |
| `/api/sku-forecasts` | Yes |
| `/api/holdout-forecasts` | Yes |

UI behaviors **not** covered by these tests: rendering, routing, loading/error states, Recharts, filters, view-mode toggles.

## Vitest bootstrap (if user asks to add tests)

From `frontend/`:

```bash
npm install -D vitest @testing-library/react @testing-library/jest-dom jsdom
```

Add to `package.json`:

```json
"scripts": {
  "test": "vitest run",
  "test:watch": "vitest"
}
```

Add `vitest.config.ts` with `@vitejs/plugin-react` and `environment: 'jsdom'`.

First tests to add: `components/ui.tsx` (`ErrorState` 404 vs generic error branches).
