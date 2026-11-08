import dotenv from "dotenv";
import express from "express";
import cors from "cors";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import multer from "multer";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import JSZip from "jszip";
import { insert_login_details } from "./database.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load backend/.env no matter which folder `node` was started from.
// (index.js lives in backend/server/, so .env is one level up.)
dotenv.config({ path: path.resolve(__dirname, "../.env") });

const DATA_DIR = path.resolve(__dirname, "../server-data");
const UPLOAD_DIR = path.join(DATA_DIR, "uploads");
const RESULT_DIR = path.join(DATA_DIR, "results");
const DB_FILE = path.join(DATA_DIR, "users.json");
fs.mkdirSync(UPLOAD_DIR, { recursive: true });
fs.mkdirSync(RESULT_DIR, { recursive: true });
if (!fs.existsSync(DB_FILE)) fs.writeFileSync(DB_FILE, JSON.stringify({ users: [] }, null, 2));

// Project root = the folder that contains `backend/` and `model_build/`.
// File location: SEM_analysis/backend/server/index.js -> two levels up.
const PROJECT_DIR = process.env.PROJECT_DIR
  ? path.resolve(process.env.PROJECT_DIR)
  : path.resolve(__dirname, "../..");
// Set PYTHON_PATH in backend/.env if `python` is not on your PATH
// (for example, the python.exe inside your virtual environment).
const PYTHON = process.env.PYTHON_PATH || "python";

const app = express();
app.use(cors({ origin: process.env.CLIENT_ORIGIN || "http://localhost:5173" }));
app.use(express.json({ limit: "1mb" }));

const PORT = Number(process.env.PORT || 4000);
// Generate a persistent JWT secret automatically on first startup.
// Keep this .env file private and do not delete it after users have registered.
const ENV_FILE = path.resolve(__dirname, "../.env");
let JWT_SECRET = process.env.JWT_SECRET;
if (!JWT_SECRET) {
  JWT_SECRET = crypto.randomBytes(48).toString("hex");
  fs.appendFileSync(ENV_FILE, `\nJWT_SECRET=${JWT_SECRET}\n`, { encoding: "utf8", mode: 0o600 });
  process.env.JWT_SECRET = JWT_SECRET;
  console.log("Generated and saved JWT_SECRET in backend/.env");
}
if (JWT_SECRET.length < 32) {
  console.error("JWT_SECRET must contain at least 32 characters.");
  process.exit(1);
}

const ACCESS_TTL = "30m";
const RESET_TTL = "15m";
// Per-file upload limit. Default 2 GB; override with MAX_FILE_MB in backend/.env.
const MAX_FILE_MB = Number(process.env.MAX_FILE_MB || 2048);
const MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024;
const allowedExt = new Set([".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".arrow"]);
const jobs = new Map();

function readDb() {
  return JSON.parse(fs.readFileSync(DB_FILE, "utf8"));
}
function writeDb(db) {
  fs.writeFileSync(DB_FILE, JSON.stringify(db, null, 2), { mode: 0o600 });
}
function publicUser(user) {
  return { id: user.id, username: user.username, email: user.email };
}
function makeUserAuthSecret() {
  // A distinct, random identifier/secret per account; JWTs are signed by the server-wide
  // secret and include this value as a claim. It is not the server signing key itself.
  return crypto.randomBytes(32).toString("hex");
}
// Password rules: 8-128 characters, one uppercase letter, one number, one special character.
function passwordProblem(password) {
  if (password.length < 8 || password.length > 128) return "Password must be 8–128 characters.";
  if (!/[A-Z]/.test(password)) return "Password must include at least one uppercase letter.";
  if (!/[0-9]/.test(password)) return "Password must include at least one number.";
  if (!/[^A-Za-z0-9]/.test(password)) return "Password must include at least one special character.";
  return null;
}
function signAccessToken(user) {
  return jwt.sign(
    { sub: user.id, username: user.username, authVersion: user.authVersion, userKeyId: user.userKeyId },
    JWT_SECRET,
    { expiresIn: ACCESS_TTL, issuer: "sem-analysis-portal", audience: "sem-analysis-client", jwtid: crypto.randomUUID() }
  );
}
function authRequired(req, res, next) {
  const header = req.headers.authorization || "";
  const [scheme, token] = header.split(" ");
  if (scheme !== "Bearer" || !token) return res.status(401).json({ error: "Sign in to continue." });
  try {
    const claims = jwt.verify(token, JWT_SECRET, { issuer: "sem-analysis-portal", audience: "sem-analysis-client" });
    const db = readDb();
    const user = db.users.find((u) => u.id === claims.sub);
    if (!user || user.authVersion !== claims.authVersion || user.userKeyId !== claims.userKeyId) {
      return res.status(401).json({ error: "Your session has expired. Please sign in again." });
    }
    req.user = user;
    req.tokenClaims = claims;
    next();
  } catch {
    return res.status(401).json({ error: "Your session is invalid or expired. Please sign in again." });
  }
}
function safeUserDir(root, userId) {
  const destination = path.resolve(root, userId);
  if (!destination.startsWith(path.resolve(root) + path.sep)) throw new Error("Invalid user directory.");
  fs.mkdirSync(destination, { recursive: true });
  return destination;
}

// Give every request its own job id (and therefore its own upload folder) so
// files from earlier runs are not analysed again.
function assignJobId(req, _res, next) {
  req.jobId = crypto.randomUUID();
  next();
}

const storage = multer.diskStorage({
  destination(req, file, cb) {
    try {
      const jobDir = path.join(safeUserDir(UPLOAD_DIR, req.user.id), req.jobId);
      fs.mkdirSync(jobDir, { recursive: true });
      cb(null, jobDir);
    } catch (e) { cb(e); }
  },
  filename(req, file, cb) {
    const ext = path.extname(file.originalname).toLowerCase();
    if (!allowedExt.has(ext)) return cb(new Error(`Unsupported image format: ${file.originalname}`));
    cb(null, `${crypto.randomUUID()}${ext}`);
  }
});
const upload = multer({
  storage,
  limits: { files: 30, fileSize: MAX_FILE_BYTES },
  fileFilter(req, file, cb) {
    const ext = path.extname(file.originalname).toLowerCase();
    if (!allowedExt.has(ext)) return cb(new Error(`Unsupported image format: ${file.originalname}`));
    cb(null, true);
  }
});

app.get("/api/health", (_req, res) => res.json({ ok: true }));

app.post("/api/auth/signup", async (req, res) => {
  const username = String(req.body.username || "").trim();
  const email = String(req.body.email || "").trim().toLowerCase();
  const password = String(req.body.password || "");
  if (username.length < 3 || username.length > 40) return res.status(400).json({ error: "Username must be 3–40 characters." });
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) return res.status(400).json({ error: "Enter a valid email address." });
  const pwError = passwordProblem(password);
  if (pwError) return res.status(400).json({ error: pwError });
  const db = readDb();
  if (db.users.some((u) => u.email === email || u.username.toLowerCase() === username.toLowerCase())) {
    return res.status(409).json({ error: "That email or username is already registered." });
  }
  const passwordHash = await bcrypt.hash(password, 12);

  // Keep the PostgreSQL record, but do not let a database problem hide the real error.
  try {
    await insert_login_details(username, passwordHash);
  } catch (e) {
    console.error("insert_login_details failed:", e);
    if (e.code === "23505") {
      return res.status(409).json({ error: "That email or username is already registered." });
    }
    return res.status(503).json({ error: "Server is busy right now. Please try again after some time." });
  }

  const user = {
    id: crypto.randomUUID(),
    username,
    email,
    passwordHash,
    userKeyId: crypto.randomUUID(),
    // Random per-user value for identity/session invalidation, not a client-visible signing key.
    userAuthSecret: makeUserAuthSecret(),
    authVersion: 1,
    createdAt: new Date().toISOString()
  };
  // Login and authRequired read users from users.json, so the user must be saved here.
  db.users.push(user);
  writeDb(db);

  safeUserDir(UPLOAD_DIR, user.id);
  safeUserDir(RESULT_DIR, user.id);
  res.status(201).json({ message: "Account created. Please sign in." });
});

app.post("/api/auth/login", async (req, res) => {
  const email = String(req.body.email || "").trim().toLowerCase();
  const password = String(req.body.password || "");
  const db = readDb();
  const user = db.users.find((u) => u.email === email);
  if (!user || !(await bcrypt.compare(password, user.passwordHash))) {
    return res.status(401).json({ error: "Email or password is incorrect." });
  }
  const token = signAccessToken(user);
  res.json({ token, expiresIn: 1800, user: publicUser(user) });
});

app.get("/api/auth/me", authRequired, (req, res) => res.json({ user: publicUser(req.user) }));

app.post("/api/auth/logout", authRequired, (req, res) => {
  const db = readDb();
  const user = db.users.find((u) => u.id === req.user.id);
  if (user) {
    user.authVersion += 1;
    writeDb(db);
  }
  res.json({ message: "Logged out." });
});

app.post("/api/auth/forgot-password", (req, res) => {
  const email = String(req.body.email || "").trim().toLowerCase();
  const db = readDb();
  const user = db.users.find((u) => u.email === email);
  // Do not disclose whether an email is registered. A real deployment should email this link.
  if (user) {
    const resetToken = jwt.sign(
      { sub: user.id, purpose: "password-reset", authVersion: user.authVersion, nonce: crypto.randomUUID() },
      JWT_SECRET,
      { expiresIn: RESET_TTL, issuer: "sem-analysis-portal", audience: "sem-analysis-reset" }
    );
    const resetUrl = `${process.env.CLIENT_PUBLIC_URL || "http://localhost:5173"}/?resetToken=${encodeURIComponent(resetToken)}`;
    console.log(`Development password reset link for ${email}: ${resetUrl}`);
  }
  res.json({ message: "If that email is registered, reset instructions will be sent." });
});

app.post("/api/auth/reset-password", async (req, res) => {
  const resetToken = String(req.body.resetToken || "");
  const password = String(req.body.password || "");
  const pwError = passwordProblem(password);
  if (pwError) return res.status(400).json({ error: pwError });
  let claims;
  try {
    claims = jwt.verify(resetToken, JWT_SECRET, { issuer: "sem-analysis-portal", audience: "sem-analysis-reset" });
  } catch {
    return res.status(400).json({ error: "Reset link is invalid or expired." });
  }
  if (claims.purpose !== "password-reset") return res.status(400).json({ error: "Invalid reset token." });
  const db = readDb();
  const user = db.users.find((u) => u.id === claims.sub);
  if (!user || user.authVersion !== claims.authVersion) return res.status(400).json({ error: "Reset link is invalid or expired." });
  user.passwordHash = await bcrypt.hash(password, 12);
  user.authVersion += 1;
  user.userAuthSecret = makeUserAuthSecret();
  user.userKeyId = crypto.randomUUID();
  writeDb(db);
  res.json({ message: "Password updated. Please sign in." });
});

app.post("/api/analysis", authRequired, assignJobId, upload.array("images", 30), (req, res) => {
  if (!req.files?.length) return res.status(400).json({ error: "Upload at least one SEM image." });
  const jobId = req.jobId;
  const resultDir = safeUserDir(RESULT_DIR, req.user.id);
  const inputDir = path.join(safeUserDir(UPLOAD_DIR, req.user.id), jobId);
  const job = {
    id: jobId,
    userId: req.user.id,
    status: "queued",
    progress: 0,
    message: "Queued for analysis…",
    error: null,
    resultPath: null,
    createdAt: new Date().toISOString(),
    files: req.files.map((f) => ({ path: f.path, originalName: f.originalname })),
    inputDir,
    resultDir
  };
  jobs.set(jobId, job);
  runPrediction(job); // runs in the background; the response returns immediately
  res.status(202).json({ jobId, status: job.status });
});

// Calls: model_build.sem_cosine_prediction.main(input_dir=Path, output_csv=Path)
// Paths are passed as arguments (sys.argv), never pasted into the code string.
const PY_RUNNER = [
  "import sys",
  "from pathlib import Path",
  "from model_build.sem_cosine_prediction import main",
  "main(input_dir=Path(sys.argv[1]), output_csv=Path(sys.argv[2]))"
].join("; ");

// Deletes this job's uploaded files (only needed while the model runs).
// The results CSV is kept so the Download button still works.
// NOTE: this must stay at the top level of the file so failJob() and the
// process "close" handler can both call it.
function cleanupUploads(job) {
  try {
    fs.rmSync(job.inputDir, { recursive: true, force: true });
    console.log(`[JOB ${job.id}] uploads deleted`);
  } catch (e) {
    console.error(`[JOB ${job.id}] could not delete uploads:`, e.message);
  }
}

function runPrediction(job) {
  const outputCsv = path.join(job.resultDir, `${job.id}-predictions.csv`);
  job.status = "running";
  job.progress = 10;
  job.message = "Preparing SEM images and datasets…";
  console.log(`[JOB ${job.id}] starting. input=${job.inputDir}`);

  let child;
  try {
    child = spawn(PYTHON, ["-c", PY_RUNNER, job.inputDir, outputCsv], {
      cwd: PROJECT_DIR,
      env: { ...process.env, PYTHONPATH: PROJECT_DIR, PYTHONUNBUFFERED: "1" }
    });
  } catch (e) {
    return failJob(job, e);
  }

  job.progress = 20;
  job.message = "Loading CNN, LSTM and Transformer models…";

  // The model reports no real progress, so creep towards 90% so the bar keeps moving.
  const ticker = setInterval(() => {
    if (job.status === "running" && job.progress < 90) {
      job.progress += 1;
      if (job.progress > 30) job.message = `Processing ${job.files.length} SEM file(s)…`;
    }
  }, 4000);

  let stderrTail = "";
  child.stdout.on("data", (d) => process.stdout.write(`[JOB ${job.id}] ${d}`));
  child.stderr.on("data", (d) => {
    process.stderr.write(`[JOB ${job.id}] ${d}`);
    stderrTail = (stderrTail + d.toString()).slice(-4000);
  });

  // Fires when Python cannot be started at all (for example `python` is not on PATH).
  child.on("error", (e) => {
    clearInterval(ticker);
    failJob(job, new Error(`Could not start "${PYTHON}": ${e.message}. Set PYTHON_PATH in backend/.env.`));
  });

  child.on("close", (code) => {
    clearInterval(ticker);
    if (job.status === "failed") return;
    if (code !== 0) {
      return failJob(job, new Error(`Python exited with code ${code}.\n${stderrTail}`));
    }
    if (!fs.existsSync(outputCsv)) {
      return failJob(job, new Error(
        `Prediction finished but the CSV was not created at ${outputCsv}. ` +
        `Check that model_build/sem_cosine_prediction.py uses the output_csv argument.`
      ));
    }
    job.resultPath = outputCsv;
    job.progress = 100;
    job.status = "completed";
    job.message = "SEM analysis completed successfully.";
    console.log(`[JOB ${job.id}] completed.`);
    cleanupUploads(job); // free disk space after a successful run
  });
}

function failJob(job, error) {
  console.error(`[JOB ${job.id}] failed:`, error.message);
  job.status = "failed";
  job.progress = 0;
  job.error = "Analysis could not be completed. Check the backend logs.";
  job.message = "SEM prediction failed.";
  cleanupUploads(job);
}

app.get("/api/analysis/:jobId", authRequired, (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job || job.userId !== req.user.id) return res.status(404).json({ error: "Analysis job not found." });
  res.json({ jobId: job.id, status: job.status, progress: job.progress, message: job.message, error: job.error });
});

app.get("/api/analysis/results/latest", authRequired, async (req, res) => {
  const userResultDir = safeUserDir(RESULT_DIR, req.user.id);
  const files = fs.readdirSync(userResultDir)
    .filter((name) => name.endsWith("-predictions.csv"))
    .map((name) => ({ name, fullPath: path.join(userResultDir, name), mtime: fs.statSync(path.join(userResultDir, name)).mtimeMs }))
    .sort((a, b) => b.mtime - a.mtime);
  if (!files.length) return res.status(404).json({ error: "No results are available yet. Run SEM analysis first." });
  try {
    const zip = new JSZip();
    zip.file("predictions.csv", fs.readFileSync(files[0].fullPath));
    const archive = await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" });
    res.setHeader("Content-Type", "application/zip");
    res.setHeader("Content-Disposition", 'attachment; filename="sem-analysis-results.zip"');
    res.end(archive);
  } catch (e) {
    console.error(e);
    if (!res.headersSent) res.status(500).json({ error: "Could not create result ZIP." });
  }
});

// Error handler: must come after all routes and before app.listen.
app.use((err, req, res, _next) => {
  console.error(err);

  // A failed upload leaves partial files behind, so remove this request's job folder.
  if (req.jobId && req.user) {
    try {
      fs.rmSync(path.join(UPLOAD_DIR, req.user.id, req.jobId), { recursive: true, force: true });
    } catch { /* ignore cleanup errors */ }
  }

  if (err instanceof multer.MulterError) {
    if (err.code === "LIMIT_FILE_SIZE") {
      return res.status(413).json({ error: `A file is larger than the ${MAX_FILE_MB} MB limit.` });
    }
    if (err.code === "LIMIT_FILE_COUNT") {
      return res.status(400).json({ error: "Too many files. Upload no more than 30 at once." });
    }
    return res.status(400).json({ error: err.message });
  }
  res.status(err.status || 500).json({ error: err.message || "Unexpected server error." });
});

app.listen(PORT, () => console.log(`SEM API listening on http://localhost:${PORT}`));