import express from "express";
import cors from "cors";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import multer from "multer";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import JSZip from "jszip";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DATA_DIR = path.resolve(__dirname, "../server-data");
const UPLOAD_DIR = path.join(DATA_DIR, "uploads");
const RESULT_DIR = path.join(DATA_DIR, "results");
const DB_FILE = path.join(DATA_DIR, "users.json");
fs.mkdirSync(UPLOAD_DIR, { recursive: true });
fs.mkdirSync(RESULT_DIR, { recursive: true });
if (!fs.existsSync(DB_FILE)) fs.writeFileSync(DB_FILE, JSON.stringify({ users: [] }, null, 2));

const app = express();
app.use(cors({ origin: process.env.CLIENT_ORIGIN || "http://localhost:5173" }));
app.use(express.json({ limit: "1mb" }));

const PORT = Number(process.env.PORT || 4000);
const JWT_SECRET = process.env.JWT_SECRET;
if (!JWT_SECRET || JWT_SECRET.length < 32) {
  console.error("Set JWT_SECRET to a random secret of at least 32 characters before starting the API.");
  process.exit(1);
}

const ACCESS_TTL = "30m";
const RESET_TTL = "15m";
const allowedExt = new Set([".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]);
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
const storage = multer.diskStorage({
  destination(req, file, cb) {
    try { cb(null, safeUserDir(UPLOAD_DIR, req.user.id)); } catch (e) { cb(e); }
  },
  filename(req, file, cb) {
    const ext = path.extname(file.originalname).toLowerCase();
    if (!allowedExt.has(ext)) return cb(new Error(`Unsupported image format: ${file.originalname}`));
    cb(null, `${crypto.randomUUID()}${ext}`);
  }
});
const upload = multer({
  storage,
  limits: { files: 30, fileSize: 25 * 1024 * 1024 },
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
  if (password.length < 8 || password.length > 128) return res.status(400).json({ error: "Password must be 8–128 characters." });
  const db = readDb();
  if (db.users.some((u) => u.email === email || u.username.toLowerCase() === username.toLowerCase())) {
    return res.status(409).json({ error: "That email or username is already registered." });
  }
  const user = {
    id: crypto.randomUUID(),
    username,
    email,
    passwordHash: await bcrypt.hash(password, 12),
    userKeyId: crypto.randomUUID(),
    // Random per-user value for identity/session invalidation, not a client-visible signing key.
    userAuthSecret: makeUserAuthSecret(),
    authVersion: 1,
    createdAt: new Date().toISOString()
  };
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

// app.post("/api/auth/reset-password", async (req, res) => {
//   const resetToken = String(req.body.resetToken || "");
//   const password = String(req.body.password || "");
//   if (password.length < 8 || password.length > 128) return res.status(400).json({ error: "Password must be 8–128 characters." });
//   let claims;
//   try {
//     claims = jwt.verify(resetToken, JWT_SECRET, { issuer: "sem-analysis-portal", audience: "sem-analysis-reset" });
//   } catch {
//     return res.status(400).json({ error: "Reset link is invalid or expired." });
//   }
//   if (claims.purpose !== "password-reset") return res.status(400).json({ error: "Invalid reset token." });
//   const db = readDb();
//   const user = db.users.find((u) => u.id === claims.sub);
//   if (!user || user.authVersion !== claims.authVersion) return res.status(400).json({ error: "Reset link is invalid or expired." });
//   user.passwordHash = await bcrypt.hash(password, 12);
//   user.authVersion += 1;
//   user.userAuthSecret = makeUserAuthSecret();
//   user.userKeyId = crypto.randomUUID();
//   writeDb(db);
//   res.json({ message: "Password updated. Please sign in." });
// });

app.post("/api/auth/forgot-password", async (req, res) => {
  try {
    const { email } = req.body;

    if (!email || typeof email !== "string") {
      return res.status(400).json({
        error: "Please enter your email address.",
      });
    }

    const normalizedEmail = email.trim().toLowerCase();

    // Find the registered user
    const users = await readUsers();
    const user = users.find(
      (u) => u.email.toLowerCase() === normalizedEmail
    );

    // Avoid revealing whether an email is registered
    if (!user) {
      return res.json({
        message: "If the account exists, a reset link will be generated.",
      });
    }

    // Generate a reset token that expires in 15 minutes
    const resetToken = jwt.sign(
      {
        userId: user.id,
        purpose: "password-reset",
      },
      JWT_SECRET,
      { expiresIn: "15m" }
    );

    const resetUrl =
      `http://localhost:5173/?resetToken=${encodeURIComponent(resetToken)}`;

    console.log("Password reset link:", resetUrl);

    return res.json({
      message: "Reset link generated. Check the backend terminal.",
    });
  } catch (error) {
    console.error("Forgot password error:", error);
    return res.status(500).json({
      error: "Unable to process the password reset request.",
    });
  }
});

app.post("/api/analysis", authRequired, upload.array("images", 30), (req, res) => {
  if (!req.files?.length) return res.status(400).json({ error: "Upload at least one SEM image." });
  const jobId = crypto.randomUUID();
  const resultDir = safeUserDir(RESULT_DIR, req.user.id);
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
    resultDir
  };
  jobs.set(jobId, job);
  void runPredictionPlaceholder(job);
  res.status(202).json({ jobId, status: job.status });
});

async function runPredictionPlaceholder(job) {
  try {
    job.status = "running";
    job.message = "Preparing images for analysis…";
    job.progress = 8;
    // Placeholder stages make the progress UI testable. Replace this function body with
    // a real worker/API call and report genuine stage progress from that worker.
    await delay(450);
    job.progress = 25; job.message = "Loading analysis model…";
    await delay(550);
    job.progress = 52; job.message = `Processing ${job.files.length} SEM image(s)…`;
    await delay(700);
    job.progress = 78; job.message = "Collecting predictions…";
    await delay(400);
    const rows = job.files.map((f) => ({
      filename: f.originalName,
      status: "placeholder",
      prediction: "Connect your prediction script here",
      confidence: null
    }));
    const outputPath = path.join(job.resultDir, `${job.id}-results.json`);
    fs.writeFileSync(outputPath, JSON.stringify({ jobId: job.id, generatedAt: new Date().toISOString(), note: "Placeholder output: no model prediction has been run.", results: rows }, null, 2));
    job.resultPath = outputPath;
    job.progress = 100;
    job.status = "completed";
    job.message = "Placeholder job complete. Connect the real prediction model to produce predictions.";
  } catch (e) {
    job.status = "failed";
    job.error = "Analysis job failed.";
    job.message = job.error;
    console.error(e);
  }
}
function delay(ms) { return new Promise((resolve) => setTimeout(resolve, ms)); }

app.get("/api/analysis/:jobId", authRequired, (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job || job.userId !== req.user.id) return res.status(404).json({ error: "Analysis job not found." });
  res.json({ jobId: job.id, status: job.status, progress: job.progress, message: job.message, error: job.error });
});

app.get("/api/analysis/results/latest", authRequired, async (req, res) => {
  const userResultDir = safeUserDir(RESULT_DIR, req.user.id);
  const files = fs.readdirSync(userResultDir)
    .filter((name) => name.endsWith("-results.json"))
    .map((name) => ({ name, fullPath: path.join(userResultDir, name), mtime: fs.statSync(path.join(userResultDir, name)).mtimeMs }))
    .sort((a, b) => b.mtime - a.mtime);
  if (!files.length) return res.status(404).json({ error: "No results are available yet. Run SEM analysis first." });
  const resultJson = JSON.parse(fs.readFileSync(files[0].fullPath, "utf8"));
  res.setHeader("Content-Type", "application/zip");
  res.setHeader("Content-Disposition", 'attachment; filename="sem-analysis-results.zip"');
  try {
    const zip = new JSZip();
    zip.file("results.json", JSON.stringify(resultJson, null, 2));
    const archive = await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" });
    res.end(archive);
  } catch (e) {
    console.error(e);
    if (!res.headersSent) res.status(500).json({ error: "Could not create result ZIP." });
  }
});

app.use((err, _req, res, _next) => {
  console.error(err);
  const status = err instanceof multer.MulterError ? 400 : 500;
  res.status(status).json({ error: err.message || "Unexpected server error." });
});

app.listen(PORT, () => console.log(`SEM API listening on http://localhost:${PORT}`));
