import React, { useEffect, useRef, useState } from "react";

const API = "/api";
const ACCEPTED = ".png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp,.arrow";
const MAX_POLL_FAILURES = 5;

const isAcceptedFile = (file) => /\.(png|jpe?g|tiff?|bmp|webp|arrow)$/i.test(file.name);

async function apiRequest(path, options = {}, token = null) {
  const headers = new Headers(options.headers || {});
  if (token) headers.set("Authorization", `Bearer ${token}`);
  // Let the browser set multipart Content-Type (including boundary) for FormData.
  if (options.body && !(options.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  let response;
  try {
    response = await fetch(`${API}${path}`, { ...options, headers });
  } catch (err) {
    console.error("Network error:", path, err);
    throw new Error("Server is busy right now. Please try again after some time.");
  }

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    console.error("API error:", path, response.status, data);
    if (response.status === 503 || response.status >= 500) {
      throw new Error("Server is busy right now. Please try again after some time.");
    }
    throw new Error(data.detail || data.error || "Request failed. Please try again.");
  }
  return data;
}

export default function App() {
  const [mode, setMode] = useState("login");
  const [token, setToken] = useState(() => sessionStorage.getItem("sem_access_token") || "");
  const [user, setUser] = useState(null);
  const [form, setForm] = useState({ username: "", email: "", password: "", confirmPassword: "" });
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [files, setFiles] = useState([]);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [jobMessage, setJobMessage] = useState("");
  const [resultReady, setResultReady] = useState(false);
  const fileInput = useRef(null);
  const pollRef = useRef(null);

  function setField(name, value) {
    setForm((prev) => ({ ...prev, [name]: value }));
  }

  function addFiles(fileList) {
    const incoming = Array.from(fileList || []);
    const accepted = incoming.filter(isAcceptedFile);
    const rejected = incoming.length - accepted.length;
    if (rejected > 0) {
      setError(`${rejected} file${rejected === 1 ? " was" : "s were"} skipped. Supported formats: PNG, JPG, TIFF, BMP, WEBP and .arrow.`);
    } else {
      setError("");
    }
    if (accepted.length) setFiles((old) => [...old, ...accepted]);
    setResultReady(false);
  }

  useEffect(() => {
    if (!token) {
      setUser(null);
      return;
    }
    apiRequest("/auth/me", {}, token)
      .then((data) => setUser(data.user))
      .catch((requestError) => {
        // A 401 means the session is invalid. A temporary server outage should
        // not erase the user's saved session token.
        if (requestError.message.includes("Server is busy")) {
          setError(requestError.message);
          return;
        }
        sessionStorage.removeItem("sem_access_token");
        setToken("");
        setUser(null);
      });
  }, [token]);

  useEffect(() => () => {
    if (pollRef.current) clearInterval(pollRef.current);
  }, []);

  async function submitAuth(event) {
    event.preventDefault();
    setError("");
    setNotice("");
    try {
      if (mode === "signup") {
        if (form.password !== form.confirmPassword) throw new Error("Passwords do not match.");
        const data = await apiRequest("/auth/signup", {
          method: "POST",
          body: JSON.stringify({ username: form.username, email: form.email, password: form.password })
        });
        setNotice(data.message || "Account created. Please sign in.");
        setMode("login");
        setForm((p) => ({ ...p, password: "", confirmPassword: "" }));
      } else if (mode === "login") {
        const data = await apiRequest("/auth/login", {
          method: "POST",
          body: JSON.stringify({ email: form.email, password: form.password })
        });
        sessionStorage.setItem("sem_access_token", data.token);
        setToken(data.token);
        setUser(data.user);
        setNotice("Signed in successfully.");
      } else if (mode === "forgot") {
        const data = await apiRequest("/auth/forgot-password", {
          method: "POST",
          body: JSON.stringify({ email: form.email })
        });
        setNotice(data.message || "If that email exists, reset instructions are available.");
      } else if (mode === "reset") {
        const resetToken = new URLSearchParams(window.location.search).get("resetToken");
        if (!resetToken) throw new Error("Reset token is missing from the URL.");
        const data = await apiRequest("/auth/reset-password", {
          method: "POST",
          body: JSON.stringify({ resetToken, password: form.password })
        });
        setNotice(data.message || "Password reset. Please sign in.");
        setMode("login");
      }
    } catch (e) {
      setError(e.message);
    }
  }

  async function logout() {
    if (pollRef.current) clearInterval(pollRef.current);
    // Best-effort server-side invalidation; the local session is cleared even if the API is unreachable.
    if (token) {
      try { await apiRequest("/auth/logout", { method: "POST" }, token); } catch { /* clear local session below */ }
    }
    sessionStorage.removeItem("sem_access_token");
    setToken("");
    setUser(null);
    setFiles([]);
    setProgress(0);
    setBusy(false);
    setResultReady(false);
    setJobMessage("");
    setMode("login");
    setNotice("You have been logged out.");
  }

  async function runAnalysis(event) {
    event.preventDefault();
    setError("");
    setNotice("");
    if (!files.length) {
      setError("Choose at least one SEM image first.");
      return;
    }
    setBusy(true);
    setProgress(3);
    setJobMessage("Uploading images and preparing the analysis job…");
    setResultReady(false);
    try {
      const body = new FormData();
      files.forEach((file) => body.append("images", file));
      const started = await apiRequest("/analysis", { method: "POST", body }, token);
      console.log("Job started:", started);

      const jobId = started && started.jobId;
      if (!jobId) {
        throw new Error("The server did not return a job ID. Check the backend logs.");
      }
      setProgress(8);
      setJobMessage("Analysis is running. You can keep this page open while it works.");

      if (pollRef.current) clearInterval(pollRef.current);
      let failures = 0;
      pollRef.current = setInterval(async () => {
        try {
          const status = await apiRequest(`/analysis/${encodeURIComponent(jobId)}`, {}, token);
          failures = 0;
          setProgress(Math.max(8, Math.min(99, Number(status.progress || 0))));
          setJobMessage(status.message || "Processing SEM images…");
          if (status.status === "completed") {
            clearInterval(pollRef.current);
            pollRef.current = null;
            setProgress(100);
            setBusy(false);
            setResultReady(true);
            setJobMessage("Analysis completed. Your results are ready to download.");
          } else if (status.status === "failed") {
            clearInterval(pollRef.current);
            pollRef.current = null;
            setBusy(false);
            setError(status.error || "The analysis failed.");
            setJobMessage("Analysis could not be completed.");
          }
        } catch (e) {
          // Tolerate a few failed polls (brief network blips) before giving up.
          failures += 1;
          if (failures >= MAX_POLL_FAILURES) {
            clearInterval(pollRef.current);
            pollRef.current = null;
            setBusy(false);
            setError(e.message);
            setJobMessage("");
          }
        }
      }, 1200);
    } catch (e) {
      setBusy(false);
      setProgress(0);
      setJobMessage("");
      setError(e.message);
    }
  }

  async function downloadResults() {
    setError("");
    try {
      const response = await fetch(`${API}/analysis/results/latest`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.detail || data.error || "No results are available yet.");
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "sem-analysis-results.zip";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(e.message);
    }
  }

  const isAuthenticated = Boolean(token && user);
  const resetTokenInUrl = new URLSearchParams(window.location.search).has("resetToken");
  useEffect(() => {
    if (!token && resetTokenInUrl) setMode("reset");
  }, [token, resetTokenInUrl]);

  if (!isAuthenticated) {
    const title = mode === "signup" ? "Create your account" :
      mode === "forgot" ? "Reset your password" :
      mode === "reset" ? "Choose a new password" : "Welcome back";
    return (
      <main className="auth-shell">
        <section className="auth-aside">
          <div className="brand"><span className="brand-mark">SEM</span><span>Insight Lab</span></div>
          <div className="aside-copy">
            <p className="eyebrow">MICROSCOPY • MACHINE LEARNING</p>
            <h1>Turn micrographs into meaningful insights.</h1>
            <p>Upload your scanning electron microscope images, run an analysis job, and retrieve results from one workspace.</p>
          </div>
          <div className="aside-footer">A workspace for your SEM research</div>
        </section>
        <section className="auth-panel">
          <div className="auth-card">
            <div className="mobile-brand brand"><span className="brand-mark">SEM</span><span>Insight Lab</span></div>
            <p className="eyebrow">SECURE WORKSPACE</p>
            <h2>{title}</h2>
            <p className="muted">{mode === "signup" ? "Set up an account to keep your files and results private." :
              mode === "forgot" ? "Enter your account email to request a password reset." :
              mode === "reset" ? "Use a strong password you have not used elsewhere." : "Sign in to continue to your SEM workspace."}</p>
            {notice && <div className="notice">{notice}</div>}
            {error && <div className="error">{error}</div>}
            <form onSubmit={submitAuth} className="auth-form">
              {mode === "signup" && <label>Username<input autoComplete="username" value={form.username} onChange={(e) => setField("username", e.target.value)} required minLength={3} /></label>}
              {mode !== "reset" && <label>Email address<input type="email" autoComplete="email" value={form.email} onChange={(e) => setField("email", e.target.value)} required /></label>}
              {(mode === "login" || mode === "signup") && <label>Password<input type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} value={form.password} onChange={(e) => setField("password", e.target.value)} required minLength={8} /></label>}
              {mode === "signup" && <label>Confirm password<input type="password" autoComplete="new-password" value={form.confirmPassword} onChange={(e) => setField("confirmPassword", e.target.value)} required minLength={8} /></label>}
              {mode === "reset" && <label>New password<input type="password" autoComplete="new-password" value={form.password} onChange={(e) => setField("password", e.target.value)} required minLength={8} /></label>}
              <button className="primary-btn" type="submit">{mode === "signup" ? "Create account" : mode === "forgot" ? "Request reset" : mode === "reset" ? "Update password" : "Sign in"} <span>→</span></button>
            </form>
            <div className="auth-links">
              {mode === "login" && <><button onClick={() => { setMode("forgot"); setError(""); setNotice(""); }}>Forgot password?</button><p>New here? <button onClick={() => { setMode("signup"); setError(""); setNotice(""); }}>Create an account</button></p></>}
              {mode === "signup" && <p>Already registered? <button onClick={() => { setMode("login"); setError(""); setNotice(""); }}>Sign in</button></p>}
              {(mode === "forgot" || mode === "reset") && <p><button onClick={() => { setMode("login"); setError(""); setNotice(""); }}>Back to sign in</button></p>}
            </div>
            <p className="tiny-note">Each account receives its own server-generated authentication identity. Never share your token.</p>
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark">SEM</span><span>Insight Lab</span></div>
        <div className="user-area"><span className="online-dot" /> <span>{user.username}</span><span className="user-chip">{user.email}</span></div>
      </header>
      <div className="page-wrap">
        <div className="page-heading">
          <div><p className="eyebrow">ANALYSIS WORKSPACE</p><h1>SEM image analysis</h1><p className="muted">Upload micrographs and run your analysis in a private workspace.</p></div>
          <div className="secure-pill"><span>●</span> Authenticated session</div>
        </div>
        {notice && <div className="notice">{notice}</div>}
        {error && <div className="error">{error}</div>}
        <section className="workspace-grid">
          <div className="panel upload-panel">
            <div className="panel-heading"><div><span className="step-number">01</span><h2>Upload SEM images</h2></div><span className="subtle-tag">MULTI-FILE</span></div>
            <p className="muted">Supported formats: PNG, JPG, TIFF, BMP, WEBP and Apache Arrow (.arrow)</p>
            <div
              className={`dropzone ${files.length ? "has-files" : ""}`}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                addFiles(e.dataTransfer.files);
              }}
            >
              <div className="upload-icon">↑</div>
              <h3>Drop your micrographs here</h3>
              <p>or choose image files from your computer</p>
              <button type="button" className="secondary-btn" onClick={() => fileInput.current?.click()}>Browse images</button>
              <input
                ref={fileInput}
                type="file"
                accept={ACCEPTED}
                multiple
                hidden
                onChange={(e) => {
                  addFiles(e.target.files);
                  e.target.value = "";
                }}
              />
            </div>
            <div className="file-summary"><span>{files.length} file{files.length === 1 ? "" : "s"} selected</span>{files.length > 0 && <button className="text-btn" type="button" onClick={() => { setFiles([]); setResultReady(false); }}>Clear all</button>}</div>
            {files.length > 0 && <ul className="file-list">{files.map((file, i) => <li key={`${file.name}-${i}`}><span className="file-type">{file.name.toLowerCase().endsWith(".arrow") ? "DATA" : "IMG"}</span><span className="file-name">{file.name}<small>{(file.size / (1024 * 1024)).toFixed(2)} MB</small></span><button type="button" aria-label={`Remove ${file.name}`} onClick={() => setFiles((old) => old.filter((_, j) => j !== i))}>×</button></li>)}</ul>}
            <button type="button" className="primary-btn run-btn" disabled={busy || !files.length} onClick={runAnalysis}>{busy ? "Analysis in progress…" : "Run SEM analysis"} <span>↗</span></button>
            <p className="privacy-note"><span>♢</span> Files are scoped to your account. Other users cannot access your results through this interface.</p>
          </div>
          <div className="panel status-panel">
            <div className="panel-heading"><div><span className="step-number">02</span><h2>Job status</h2></div><span className={`status-tag ${busy ? "working" : resultReady ? "complete" : ""}`}>{busy ? "RUNNING" : resultReady ? "COMPLETE" : "IDLE"}</span></div>
            <div className="status-illustration"><div className="scan-ring ring-one" /><div className="scan-ring ring-two" /><div className="scan-core"><span>SEM</span><small>ANALYSIS</small></div></div>
            <h3>{busy ? "Working on your images" : resultReady ? "Your results are ready" : "Ready when you are"}</h3>
            <p className="muted status-description">{jobMessage || "Start an analysis to see live progress and status updates here."}</p>
            {(busy || resultReady) && <div className="progress-area"><div className="progress-label"><span>{busy ? "Processing" : "Finished"}</span><strong>{progress}%</strong></div><div className="progress-track" role="progressbar" aria-valuenow={progress} aria-valuemin="0" aria-valuemax="100"><div className="progress-fill" style={{ width: `${progress}%` }} /></div><p className="progress-hint">{busy ? "This may take a while for large batches. The page will update automatically." : "The job completed successfully."}</p></div>}
            <div className="status-divider" />
            <div className="status-facts"><div><span className="fact-icon">◉</span><span><strong>Private workspace</strong><small>Results linked to your account</small></span></div><div><span className="fact-icon">↻</span><span><strong>Progress updates</strong><small>Automatic status polling</small></span></div></div>
          </div>
        </section>
        <footer className="bottom-actions"><div><p className="eyebrow">FINISHING UP</p><strong>When your analysis is complete</strong><p className="muted">Supported formats: PNG, JPG, TIFF, BMP, WEBP and Apache Arrow (.arrow). Download the result bundle or safely end your session.</p></div><div className="action-buttons"><button className="secondary-btn download-btn" onClick={downloadResults}>↓ Download results ZIP</button><button className="logout-btn" onClick={logout}>↗ Log out</button></div></footer>
        <p className="footer-note">SEM Insight Lab · Keep your authentication token private</p>
      </div>
    </main>
  );
}