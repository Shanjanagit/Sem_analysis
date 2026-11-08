# SEM Analysis Portal (React + Express + JWT)

This starter contains a React/Vite frontend and an Express API. It includes signup, login, forgot/reset-password flow, per-user identity keys, JWT authentication, per-user upload/result directories, upload UI, analysis progress polling, result ZIP download, and logout.

**Important:** the prediction function is a placeholder. It does not run your SEM model yet; it generates a JSON file that clearly marks predictions as placeholders. Replace `runPredictionPlaceholder()` in `server/index.js` with your actual model worker/API integration.

## Requirements
- Node.js 18+
- Node.js 18+ (the result ZIP is created with JSZip).

## Install
```bash
npm install
```

## Start on Windows PowerShell

Terminal 1 — API:
```powershell
$env:JWT_SECRET = [Convert]::ToHexString([Security.Cryptography.RandomNumberGenerator]::GetBytes(48))
$env:CLIENT_ORIGIN = "http://localhost:5173"
$env:CLIENT_PUBLIC_URL = "http://localhost:5173"
npm run server
```

Terminal 2 — React:
```powershell
npm run dev
```

Open `http://localhost:5173`.

## JWT and user-specific key design
- Every signup creates a cryptographically random `userAuthSecret` and unique `userKeyId` on the server.
- The client never receives the per-user secret.
- The API signs JWTs with the server-wide `JWT_SECRET`; each JWT contains the user's ID, auth version, and per-user key ID. The API checks those values against the stored user record on every protected request.
- This is safer than sending a user's "secret key" to the browser. If the per-user key ID rotates or the auth version changes, prior tokens become invalid.
- Access tokens expire after 30 minutes and are held in `sessionStorage`, which is cleared on logout or when the tab/session ends. For a production deployment, consider secure HttpOnly cookies, CSRF defenses, rate limits, HTTPS, a real database, and token revocation/session records.

## Forgot password
For development, the reset link is printed in the API terminal. Visit the URL shown there. No email is sent by this starter. In production, deliver a single-use reset link through an email provider and rate-limit requests.

## Current limits
- Uses a JSON file as a lightweight demo user store; it is not appropriate for multiple production server instances or high-concurrency writes. Use PostgreSQL for a real multi-user deployment.
- Job status is held in process memory; a server restart loses active job status. Use Redis/database-backed jobs for production.
- Progress percentages are placeholder stages to demonstrate UI feedback. The real prediction worker should report actual progress.
- Add rate limiting, CSRF/security headers, file-content validation, malware scanning where appropriate, cleanup/retention policy, and HTTPS before public deployment.
