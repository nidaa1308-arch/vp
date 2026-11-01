# ParkVision
1. `mysql -u <user> -p < backend/schema.sql`
2. `cd backend && cp .env.example .env` (fill in values) `&& npm install && npm start`
3. Open http://localhost:3000 and register. Make an admin: `npm run make-admin -- you@example.com`
Hardware ingest: `POST /api/iot/observation` with header `x-device-key` and JSON `{slot_code, source: "camera"|"sensor", status: "free"|"occupied", confidence}`.
Reconciliation: latest reading wins; if camera and sensor disagree within 30 s the slot becomes `unknown`.
