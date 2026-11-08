import os
import uuid
import requests
from dotenv import load_dotenv
# from logs import logger
import logging

load_dotenv()

logger = logging.getLogger("smoke_test")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"  # Adds human-readable time format
    )

BASE_URL = os.getenv("API_URL", "http://127.0.0.1:4000")

def run_smoke_test():
    logger.info("\n========== SEM BACKEND SMOKE TEST ==========\n")

    # 1. Check server and database
    logger.info("1. Checking server and database...")

    try:
        health = requests.get(
            f"{BASE_URL}/api/health",
            timeout=8
        )

        if health.status_code != 200:
            logger.info("FAILED: Server or database is unavailable.")
            logger.info("Response:", health.text)
            return

        logger.info("PASSED:", health.json())

    except requests.RequestException:
        logger.info("FAILED: Cannot connect to backend.")
        logger.info("Start the FastAPI server and try again.")
        return

    # 2. Create a unique test account
    logger.info("\n2. Testing signup...")

    unique_id = uuid.uuid4().hex[:10]

    test_username = f"smoketest_{unique_id}"
    test_email = f"smoketest_{unique_id}@example.com"
    test_password = "SmokeTest@123"

    try:
        signup_response = requests.post(
            f"{BASE_URL}/api/auth/signup",
            json={
                "username": test_username,
                "email": test_email,
                "password": test_password
            },
            timeout=8
        )

        if signup_response.status_code != 200:
            logger.info("FAILED: Signup did not complete.")
            logger.info("Response:", signup_response.text)
            return

        logger.info("PASSED: Test account created.")

    except requests.RequestException as error:
        logger.info("FAILED: Signup request error:", error)
        return

    # 3. Test login
    logger.info("\n3. Testing login...")

    try:
        login_response = requests.post(
            f"{BASE_URL}/api/auth/login",
            json={
                "email": test_email,
                "password": test_password
            },
            timeout=8
        )

        if login_response.status_code != 200:
            logger.info("FAILED: Login unsuccessful.")
            logger.info("Response:", login_response.text)
            return

        login_data = login_response.json()
        token = login_data.get("access_token")

        if not token:
            logger.info("FAILED: Login response did not contain a token.")
            return

        logger.info("PASSED: Login successful. JWT received.")

    except requests.RequestException as error:
        logger.info("FAILED: Login request error:", error)
        return

    # 4. Test protected dashboard
    logger.info("\n4. Testing dashboard authentication...")

    headers = {
        "Authorization": f"Bearer {token}"
    }

    try:
        dashboard_response = requests.get(
            f"{BASE_URL}/api/dashboard",
            headers=headers,
            timeout=8
        )

        if dashboard_response.status_code != 200:
            logger.info("FAILED: Dashboard access denied.")
            logger.info("Response:", dashboard_response.text)
            return

        logger.info("PASSED:", dashboard_response.json())

    except requests.RequestException as error:
        logger.info("FAILED: Dashboard request error:", error)
        return

    # 5. Check that an incorrect password is rejected
    logger.info("\n5. Testing invalid password...")

    try:
        invalid_response = requests.post(
            f"{BASE_URL}/api/auth/login",
            json={
                "email": test_email,
                "password": "WrongPassword@123"
            },
            timeout=8
        )

        if invalid_response.status_code != 401:
            logger.info("FAILED: Incorrect password was not rejected.")
            logger.info("Response:", invalid_response.text)
            return

        logger.info("PASSED: Incorrect password rejected.")

    except requests.RequestException as error:
        logger.info("FAILED: Invalid-login test error:", error)
        return

    # 6. Test dashboard without token
    logger.info("\n6. Testing dashboard without authentication...")

    try:
        protected_response = requests.get(
            f"{BASE_URL}/api/dashboard",
            timeout=8
        )

        if protected_response.status_code != 401:
            logger.info("FAILED: Unauthenticated dashboard access was not rejected.")
            return

        logger.info("PASSED: Unauthenticated access rejected.")

    except requests.RequestException as error:
        logger.info("FAILED: Protected-route test error:", error)
        return

    logger.info("\n============================================")
    logger.info("ALL SMOKE TESTS PASSED!")
    logger.info("Backend, database, signup, login and dashboard are working.")
    logger.info("============================================\n")


if __name__ == "__main__":
    run_smoke_test()