from __future__ import annotations

import asyncio
import os
import secrets
import shutil
import uuid
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import uvicorn
import bcrypt
import psycopg2
from psycopg2 import errors
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr, Field

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
load_dotenv(BASE_DIR / ".env")

# Get project root directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Get model directory
model_path = PROJECT_DIR / "model_build"

print(f"Project path: {PROJECT_DIR}")
print(f"Model path: {model_path}")

# Add project root to Python import path
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Import prediction module
from model_build.sem_cosine_prediction import main as run_sem_prediction

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "demodb")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_MINUTES = int(os.getenv("ACCESS_TOKEN_MINUTES", "30"))
RESET_TOKEN_MINUTES = 15
CLIENT_ORIGIN = os.getenv("CLIENT_ORIGIN", "http://localhost:5173")
CLIENT_PUBLIC_URL = os.getenv("CLIENT_PUBLIC_URL", CLIENT_ORIGIN)

if len(JWT_SECRET) < 32:
    raise RuntimeError(
        "JWT_SECRET must be set in backend/.env and contain at least 32 characters."
    )

DATA_DIR = PROJECT_DIR / "server-data"
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".arrow"}
MAX_FILE_BYTES = 500 * 1024 * 1024  # 500 MB
MAX_FILES = 30

app = FastAPI(title="SEM Insight Lab API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[CLIENT_ORIGIN],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# In-memory job status is suitable for local development. Use a persistent queue
# (for example Celery/RQ) before running multiple production workers.
jobs: dict[str, dict] = {}


# ------------------------------------------------------------
# Database
# ------------------------------------------------------------

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=5,
    )


def initialize_database() -> None:
    """Create the users table and add columns needed by this application."""
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id BIGSERIAL PRIMARY KEY,
                    username VARCHAR(100) NOT NULL,
                    email VARCHAR(255),
                    password_hash TEXT NOT NULL,
                    auth_version INTEGER NOT NULL DEFAULT 1,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            # Safe additions for an existing users table created by the earlier
            # Node/PostgreSQL code. Existing accounts without email cannot log in
            # by email until their email addresses are populated.
            cursor.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS email VARCHAR(255);")
            cursor.execute(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS auth_version INTEGER NOT NULL DEFAULT 1;"
            )
            cursor.execute(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP;"
            )
            cursor.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS users_email_lower_uidx "
                "ON users (LOWER(email)) WHERE email IS NOT NULL;"
            )
            cursor.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS users_username_lower_uidx "
                "ON users (LOWER(username));"
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def database_available() -> bool:
    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1;")
            cursor.fetchone()
        return True
    except psycopg2.Error:
        return False
    finally:
        if connection:
            connection.close()


@app.on_event("startup")
def on_startup():
    try:
        initialize_database()
        print("SEM Insight Lab: PostgreSQL is ready.")
    except Exception as exc:
        # Keep the API process alive so the frontend can show the friendly
        # service-unavailable message and the health check can report 503.
        print(f"Database startup check failed: {exc}")


# ------------------------------------------------------------
# Request models
# ------------------------------------------------------------

class SignupRequest(BaseModel):
    username: str = Field(min_length=3, max_length=40)
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=72)


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    resetToken: str = Field(min_length=20)
    password: str = Field(min_length=8, max_length=72)


# ------------------------------------------------------------
# Authentication helpers
# ------------------------------------------------------------

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def public_user(user: dict) -> dict:
    return {
        "id": user["id"],
        "username": user["username"],
        "email": user["email"],
    }


def create_access_token(user: dict) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user["id"]),
        "username": user["username"],
        "auth_version": user["auth_version"],
        "iss": "sem-analysis-portal",
        "aud": "sem-analysis-client",
        "iat": now,
        "exp": now + timedelta(minutes=ACCESS_TOKEN_MINUTES),
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_reset_token(user: dict) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user["id"]),
        "purpose": "password-reset",
        "auth_version": user["auth_version"],
        "iss": "sem-analysis-portal",
        "aud": "sem-analysis-reset",
        "iat": now,
        "exp": now + timedelta(minutes=RESET_TOKEN_MINUTES),
        "nonce": secrets.token_urlsafe(16),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(authorization: Optional[str] = Header(default=None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Sign in to continue.")

    token = authorization.split(" ", 1)[1].strip()
    try:
        claims = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            issuer="sem-analysis-portal",
            audience="sem-analysis-client",
        )
        user_id = int(claims["sub"])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(
            status_code=401,
            detail="Your session is invalid or expired. Please sign in again.",
        )

    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT id, username, email, auth_version FROM users WHERE id = %s",
                (user_id,),
            )
            user = cursor.fetchone()
        if not user or user["auth_version"] != claims.get("auth_version"):
            raise HTTPException(
                status_code=401,
                detail="Your session has expired. Please sign in again.",
            )
        return dict(user)
    except HTTPException:
        raise
    except psycopg2.Error:
        raise HTTPException(
            status_code=503,
            detail="Server is busy right now. Please try again after some time.",
        )
    finally:
        if connection:
            connection.close()


def user_directory(root: Path, user_id: int) -> Path:
    destination = (root / str(user_id)).resolve()
    root_resolved = root.resolve()
    if root_resolved not in destination.parents:
        raise HTTPException(status_code=400, detail="Invalid user storage location.")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def service_unavailable() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail="Server is busy right now. Please try again after some time.",
    )

def process_analysis_job(job_id: str, input_dir: Path, result_dir: Path):  # plain def
    print(f"[JOB {job_id}] started, input_dir={input_dir}")
    job = jobs.get(job_id)
    if not job:
        return
    try:
        job.update(status="running", progress=10, message="Preparing SEM images...")
        output_csv = result_dir / f"{job_id}-predictions.csv"
        job.update(progress=20, message="Loading models...")

        run_sem_prediction(input_dir=input_dir, output_csv=output_csv)

        if not output_csv.exists():
            raise RuntimeError("Prediction finished without creating the output CSV.")

        job.update(resultPath=str(output_csv), progress=100,
                   status="completed", message="SEM analysis completed successfully.")
    except Exception as exc:
        import traceback; traceback.print_exc()   # see the real error in the terminal
        job.update(status="failed", progress=0,
                   error="Analysis could not be completed. Check the backend logs.",
                   message="SEM prediction failed.")

# ------------------------------------------------------------
# Health check
# ------------------------------------------------------------

@app.get("/api/health")
def health_check():
    if not database_available():
        raise service_unavailable()
    return {"ok": True, "server": "running", "database": "connected"}


# ------------------------------------------------------------
# Signup and login
# ------------------------------------------------------------

@app.post("/api/auth/signup", status_code=201)
def signup(data: SignupRequest):
    username = data.username.strip()
    email = str(data.email).strip().lower()
    if len(username) < 3 or len(username) > 40:
        raise HTTPException(status_code=400, detail="Username must be 3–40 characters.")

    password_hash = hash_password(data.password)
    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                INSERT INTO users (username, email, password_hash)
                VALUES (%s, %s, %s)
                RETURNING id, username, email, auth_version
                """,
                (username, email, password_hash),
            )
            user = dict(cursor.fetchone())
        connection.commit()
        user_directory(UPLOAD_DIR, user["id"])
        user_directory(RESULT_DIR, user["id"])
        return {"message": "Account created successfully. Please sign in.", "user": public_user(user)}
    except errors.UniqueViolation:
        if connection:
            connection.rollback()
        raise HTTPException(status_code=409, detail="That email or username is already registered.")
    except psycopg2.Error:
        if connection:
            connection.rollback()
        raise service_unavailable()
    finally:
        if connection:
            connection.close()


@app.post("/api/auth/login")
def login(data: LoginRequest):
    email = str(data.email).strip().lower()
    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT id, username, email, password_hash, auth_version
                FROM users
                WHERE LOWER(email) = %s
                LIMIT 1
                """,
                (email,),
            )
            user = cursor.fetchone()

        if not user or not verify_password(data.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Email or password is incorrect.")

        token = create_access_token(dict(user))
        return {
            "message": "Signed in successfully.",
            "token": token,  # Matches the existing React App.jsx contract.
            "token_type": "bearer",
            "expiresIn": ACCESS_TOKEN_MINUTES * 60,
            "user": public_user(dict(user)),
        }
    except HTTPException:
        raise
    except psycopg2.Error:
        raise service_unavailable()
    finally:
        if connection:
            connection.close()


@app.get("/api/auth/me")
def get_me(current_user: dict = Depends(get_current_user)):
    return {"user": public_user(current_user)}


@app.post("/api/auth/logout")
def logout(current_user: dict = Depends(get_current_user)):
    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE users SET auth_version = auth_version + 1 WHERE id = %s",
                (current_user["id"],),
            )
        connection.commit()
        return {"message": "Logged out successfully."}
    except psycopg2.Error:
        if connection:
            connection.rollback()
        raise service_unavailable()
    finally:
        if connection:
            connection.close()


# ------------------------------------------------------------
# Forgot password / reset password
# ------------------------------------------------------------

@app.post("/api/auth/forgot-password")
def forgot_password(data: ForgotPasswordRequest):
    email = str(data.email).strip().lower()
    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT id, username, email, auth_version FROM users WHERE LOWER(email) = %s LIMIT 1",
                (email,),
            )
            user = cursor.fetchone()

        # Do not reveal whether the email address exists.
        if user:
            reset_token = create_reset_token(dict(user))
            reset_url = f"{CLIENT_PUBLIC_URL}/?resetToken={reset_token}"
            # Development-only delivery. Configure an email provider before deployment.
            print(f"\n[DEV ONLY] Password reset link for {email}:\n{reset_url}\n")
        return {"message": "If that email is registered, reset instructions will be sent."}
    except psycopg2.Error:
        raise service_unavailable()
    finally:
        if connection:
            connection.close()


@app.post("/api/auth/reset-password")
def reset_password(data: ResetPasswordRequest):
    try:
        claims = jwt.decode(
            data.resetToken,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            issuer="sem-analysis-portal",
            audience="sem-analysis-reset",
        )
        if claims.get("purpose") != "password-reset":
            raise JWTError("Wrong token purpose")
        user_id = int(claims["sub"])
        token_auth_version = int(claims["auth_version"])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(status_code=400, detail="Reset link is invalid or expired.")

    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT id, auth_version FROM users WHERE id = %s",
                (user_id,),
            )
            user = cursor.fetchone()
            if not user or user["auth_version"] != token_auth_version:
                raise HTTPException(status_code=400, detail="Reset link is invalid or expired.")
            cursor.execute(
                "UPDATE users SET password_hash = %s, auth_version = auth_version + 1 WHERE id = %s",
                (hash_password(data.password), user_id),
            )
        connection.commit()
        return {"message": "Password updated successfully. Please sign in."}
    except HTTPException:
        if connection:
            connection.rollback()
        raise
    except psycopg2.Error:
        if connection:
            connection.rollback()
        raise service_unavailable()
    finally:
        if connection:
            connection.close()


# ------------------------------------------------------------
# SEM image upload and analysis job endpoints
# ------------------------------------------------------------

async def save_upload(upload: UploadFile, destination: Path) -> dict:
    extension = Path(upload.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported image format: {upload.filename}")

    safe_name = f"{uuid.uuid4()}{extension}"
    target = destination / safe_name
    size = 0
    try:
        with target.open("wb") as output:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_FILE_BYTES:
                    raise HTTPException(status_code=413, detail=f"{upload.filename} exceeds the 25 MB limit.")
                output.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

    return {"path": str(target), "originalName": upload.filename or safe_name}


# async def run_prediction_placeholder(job_id: str):
#     job = jobs[job_id]
#     try:
#         job.update(status="running", progress=8, message="Preparing images for analysis…")
#         await asyncio.sleep(0.45)
#         job.update(progress=25, message="Loading analysis model…")
#         await asyncio.sleep(0.55)
#         job.update(progress=52, message=f"Processing {len(job['files'])} SEM image(s)…")
#         await asyncio.sleep(0.7)
#         job.update(progress=78, message="Collecting predictions…")
#         await asyncio.sleep(0.4)

#         rows = [
#             {
#                 "filename": item["originalName"],
#                 "status": "placeholder",
#                 "prediction": "Connect your SEM prediction pipeline here",
#                 "confidence": None,
#             }
#             for item in job["files"]
#         ]
#         result_path = Path(job["resultDir"]) / f"{job_id}-results.json"
#         import json
#         result_path.write_text(
#             json.dumps(
#                 {
#                     "jobId": job_id,
#                     "generatedAt": datetime.now(timezone.utc).isoformat(),
#                     "note": "Placeholder output: the real ML prediction pipeline has not been connected yet.",
#                     "results": rows,
#                 },
#                 indent=2,
#             ),
#             encoding="utf-8",
#         )
#         job.update(
#             resultPath=str(result_path),
#             progress=100,
#             status="completed",
#             message="Placeholder job complete. Connect the real prediction model to produce predictions.",
#         )
#     except Exception as exc:
#         print(f"Analysis job {job_id} failed: {exc}")
#         job.update(status="failed", error="Analysis job failed.", message="Analysis could not be completed.")

# ------------------------------------------------------------
# Real SEM prediction background job
# ------------------------------------------------------------

async def process_analysis_job(job_id: str, input_dir: Path, result_dir: Path):  # plain def
    job = jobs.get(job_id)
    if not job:
        return
    try:
        job.update(status="running", progress=10, message="Preparing SEM images...")
        output_csv = result_dir / f"{job_id}-predictions.csv"
        job.update(progress=20, message="Loading models...")

        run_sem_prediction(input_dir=input_dir, output_csv=output_csv)

        if not output_csv.exists():
            raise RuntimeError("Prediction finished without creating the output CSV.")

        job.update(resultPath=str(output_csv), progress=100,
                   status="completed", message="SEM analysis completed successfully.")
    except Exception as exc:
        import traceback; traceback.print_exc()   # see the real error in the terminal
        job.update(status="failed", progress=0,
                   error="Analysis could not be completed. Check the backend logs.",
                   message="SEM prediction failed.")

@app.post("/api/analysis", status_code=202)
async def start_analysis(
    background_tasks: BackgroundTasks,
    images: list[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
):
    if not images:
        raise HTTPException(status_code=400, detail="Upload at least one SEM image.")
    if len(images) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Upload no more than {MAX_FILES} images at once.")

    job_id = str(uuid.uuid4())
    # per-job folder so old uploads aren't re-processed
    job_upload_dir = user_directory(UPLOAD_DIR, current_user["id"]) / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    result_folder = user_directory(RESULT_DIR, current_user["id"])

    saved_files = []
    try:
        for image in images:
            saved_files.append(await save_upload(image, job_upload_dir))
    except Exception:
        shutil.rmtree(job_upload_dir, ignore_errors=True)
        raise

    jobs[job_id] = {
        "id": job_id, "userId": current_user["id"], "status": "queued",
        "progress": 0, "message": "Queued for analysis…", "error": None,
        "resultPath": None, "createdAt": datetime.now(timezone.utc).isoformat(),
        "files": saved_files, "resultDir": str(result_folder),
    }

    background_tasks.add_task(process_analysis_job, job_id, job_upload_dir, result_folder)
    return {"jobId": job_id, "status": "queued"}   # ✅ required by the frontend


@app.get("/api/analysis/{job_id}")
def get_analysis_status(job_id: str, current_user: dict = Depends(get_current_user)):
    job = jobs.get(job_id)
    if not job or job["userId"] != current_user["id"]:
        raise HTTPException(status_code=404, detail="Analysis job not found.")
    return {
        "jobId": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "error": job["error"],
    }


@app.get("/api/analysis/results/latest")
def download_latest_results(current_user: dict = Depends(get_current_user)):
    result_folder = user_directory(RESULT_DIR, current_user["id"])

    # Find this user's prediction CSVs, newest first
    result_files = sorted(
        result_folder.glob("*-predictions.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not result_files:
        raise HTTPException(
            status_code=404,
            detail="No results are available yet. Run SEM analysis first.",
        )

    zip_path = result_folder / "sem-analysis-results.zip"
    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(result_files[0], arcname="predictions.csv")
    except OSError:
        raise HTTPException(status_code=500, detail="Could not create result ZIP.")

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename="sem-analysis-results.zip",
    )

@app.get("/")
def root():
    return {"message": "SEM Insight Lab API is running."}

if __name__ == "__main__":
    print("Startting the uvicorn job run..")
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
    print(f"The uvicorn is listening in port")
