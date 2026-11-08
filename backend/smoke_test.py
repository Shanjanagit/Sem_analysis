"""End-to-end local smoke test for SEM Insight Lab FastAPI backend.

Creates one disposable test account in PostgreSQL. It does not delete that account.
Run with the backend running: python smoke_test.py
"""

import os
import uuid
import requests
from dotenv import load_dotenv
import logging
from pathlib import Path


import logging
from pathlib import Path

def get_logger(name="smoke_test"):

    # Create logs folder
    LOG_DIR = Path(__file__).resolve().parent / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Log file path
    LOG_FILE = LOG_DIR / "sem_analysis.log"

    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:

        # Log format
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        # Console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)

        # File handler
        file_handler = logging.FileHandler(
            LOG_FILE,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        logger.addHandler(console)
        logger.addHandler(file_handler)

    return logger


# Create logger
logger = get_logger()


load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 10


def fail(message, response=None):
    logger.info(f"[FAIL] {message}")
    if response is not None:
        logger.info("       HTTP", response.status_code, response.text[:500])
    raise SystemExit(1)


def main():
    logger.info("\n========== SEM INSIGHT LAB SMOKE TEST ==========")

    # 1. API + PostgreSQL health
    logger.info("1. Checking API and PostgreSQL...")
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=TIMEOUT)
    except requests.RequestException as exc:
        fail(f"Cannot reach backend at {BASE_URL}: {exc}")
    if response.status_code != 200:
        fail("Health check failed. Check FastAPI and PostgreSQL.", response)
    logger.info("PASS", response.json())

    # 2. Create a unique account
    suffix = uuid.uuid4().hex[:10]
    email = f"sem-smoke-{suffix}@example.com"
    username = f"smoke_{suffix}"
    password = "SmokeTest@12345"

    logger.info("2. Testing signup...", end=" ")
    try:
        response = requests.post(
            f"{BASE_URL}/api/auth/signup",
            json={"username": username, "email": email, "password": password},
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:
        fail(f"Signup request failed: {exc}")
    if response.status_code != 201:
        fail("Signup failed.", response)
    logger.info("PASS")

    # 3. Correct login
    logger.info("3. Testing valid login...", end=" ")
    try:
        response = requests.post(
            f"{BASE_URL}/api/auth/login",
            json={"email": email, "password": password},
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:
        fail(f"Login request failed: {exc}")
    if response.status_code != 200:
        fail("Valid login failed.", response)
    data = response.json()
    token = data.get("token")
    if not token or not data.get("user", {}).get("id"):
        fail("Login response did not include the expected token and user.", response)
    logger.info("PASS")

    # 4. Protected dashboard
    logger.info("4. Testing protected dashboard...", end=" ")
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(f"{BASE_URL}/api/auth/me", headers=headers, timeout=TIMEOUT)
    except requests.RequestException as exc:
        fail(f"Authenticated request failed: {exc}")
    if response.status_code != 200 or not response.json().get("user"):
        fail("Authenticated user endpoint failed.", response)
    logger.info("PASS")

    # 5. Reject wrong password
    logger.info("5. Testing invalid password rejection...", end=" ")
    response = requests.post(
        f"{BASE_URL}/api/auth/login",
        json={"email": email, "password": "DefinitelyWrong123"},
        timeout=TIMEOUT,
    )
    if response.status_code != 401:
        fail("Wrong password was not rejected with HTTP 401.", response)
    logger.info("PASS")

    # 6. Reject protected endpoint without token
    logger.info("6. Testing unauthenticated access rejection...", end=" ")
    response = requests.get(f"{BASE_URL}/api/auth/me", timeout=TIMEOUT)
    if response.status_code != 401:
        fail("Protected endpoint did not reject a request without a token.", response)
    logger.info("PASS")

    # 7. Logout invalidates token
    logger.info("7. Testing logout and token invalidation...", end=" ")
    response = requests.post(f"{BASE_URL}/api/auth/logout", headers=headers, timeout=TIMEOUT)
    if response.status_code != 200:
        fail("Logout failed.", response)
    response = requests.get(f"{BASE_URL}/api/auth/me", headers=headers, timeout=TIMEOUT)
    if response.status_code != 401:
        fail("Token remained valid after logout.", response)
    logger.info("PASS")

    logger.info("\n========== ALL SMOKE TESTS PASSED ==========")
    logger.info("API, database, signup, login, protected access, invalid credentials and logout work.")
    logger.info(f"A test account was created: {email}")
    logger.info("You can remove this test account from PostgreSQL when you no longer need it.\n")


if __name__ == "__main__":
    main()
