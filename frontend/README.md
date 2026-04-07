# SENTINEL/OPS — Frontend

Multimodal AI threat and anomaly detection operator dashboard.

## Stack
- **Vite** + **React 18** + **TypeScript** (strict)
- **Tailwind CSS** — design token system
- **Zustand** — UI state
- **React Query** — server/WebSocket state
- **Zod** — runtime message validation
- **pnpm** — package manager

## Quick start

```bash
# Install dependencies
pnpm install

# Terminal 1 — Start mock WebSocket server
node src/mocks/ws-server.cjs

# Terminal 2 — Start dev server
pnpm dev
```

Open http://localhost:3000

**Dev login credentials:**
- operator@sentinel.ops / sentinel2024
- analyst@sentinel.ops / sentinel2024
- admin@sentinel.ops / sentinel2024

## Project structure

```
src/
├── api/            # REST API calls (axios)
├── components/
│   ├── layout/     # AppShell, ProtectedRoute
│   ├── panels/     # Dashboard feature panels
│   └── ui/         # Primitive components
├── hooks/          # useAlertStream, useOperatorActions
├── mocks/          # ws-server.cjs — mock WebSocket server
├── pages/          # DashboardPage, LoginPage, UnauthorisedPage
├── store/          # Zustand UI store
├── types/          # TypeScript interfaces + Zod schemas
└── utils/          # Formatters, helpers, cn()
```

## Switching modes (dev)

Use the CLOUD / DEGRADED / INCIDENT toggle in the topbar.

To switch the mock server's mode, send a WebSocket message:
```json
{ "setMode": "degraded" }
```

## Connecting to a real backend

1. Update `VITE_WS_URL` in `.env` to your backend WebSocket URL
2. Ensure backend message shape matches `src/types/index.ts`
3. Remove the mode toggle from `AppShell.tsx` (mode will be set by orchestration layer)
4. Replace mock login in `LoginPage.tsx` with real Auth0/Keycloak integration

## Build for production

```bash
pnpm build
pnpm preview
```
