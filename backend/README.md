# SEM Analysis Portal — Backend

This Express API provides signup, login, logout, forgot/reset password (development reset URL is printed in the terminal), JWT-protected routes, image upload, progress polling, and ZIP result download.

## Requirements
- Node.js 18 or newer

## Install and run (Windows PowerShell)
From this `backend` folder:

```powershell
npm install
npm run dev
```

No manual JWT terminal command or pre-existing `.env` file is required. On first startup, `server/index.js` generates a cryptographically random JWT secret and writes it to `.env`. Keep that file safe and do not delete it, or existing tokens will become invalid.

The API listens on `http://localhost:4000` by default. Set `PORT`, `CLIENT_ORIGIN`, and `CLIENT_PUBLIC_URL` in `.env` if you need to customize them. The frontend should call `http://localhost:4000` unless you configure another port.

## API endpoints
- `GET /api/health`
- `POST /api/auth/signup` — JSON: `{ "username": "...", "email": "...", "password": "..." }`
- `POST /api/auth/login` — JSON: `{ "email": "...", "password": "..." }`
- `GET /api/auth/me` — requires `Authorization: Bearer <token>`
- `POST /api/auth/logout` — requires bearer token
- `POST /api/auth/forgot-password` — development reset link is logged in the server terminal; no email is sent
- `POST /api/auth/reset-password` — JSON: `{ "resetToken": "...", "password": "..." }`
- `POST /api/analysis` — requires bearer token and multipart form field `images` (up to 30 image files)
- `GET /api/analysis/:jobId` — requires bearer token
- `GET /api/analysis/results/latest` — requires bearer token; downloads ZIP

## Important limitation
The analysis worker is a placeholder and does not run a real SEM ML model. Replace `runPredictionPlaceholder()` in `server/index.js` with an integration to your actual model. This starter uses a JSON file for demo user storage and in-memory job state; use a proper database and persistent job queue for production.
