import dotenv from "dotenv";
import path from "node:path";
import { fileURLToPath } from "node:url";
import pg from "pg";

// Load backend/.env no matter which folder `node` was started from.
// (database.js lives in backend/server/, so .env is one level up.)
const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.resolve(__dirname, "../.env") });

const { Pool } = pg;

// Warn early if a required setting is missing, instead of failing later with a
// confusing "client password must be a string" error.
for (const name of ["DB_NAME", "DB_USER", "DB_PASSWORD"]) {
  if (!process.env[name]) {
    console.error(`[database] ${name} is not set. Add it to backend/.env`);
  }
}

const pool = new Pool({
  host: process.env.DB_HOST || "localhost",
  port: Number(process.env.DB_PORT || 5432),
  database: process.env.DB_NAME,
  user: process.env.DB_USER,
  // pg requires a string here; an undefined value causes the SASL error.
  password: String(process.env.DB_PASSWORD ?? ""),
});

export async function insert_login_details(username, passwordHash) {
  const result = await pool.query(
    `INSERT INTO users (username, password_hash)
     VALUES ($1, $2)
     RETURNING id`,
    [username, passwordHash]
  );

  return result.rows[0].id;
}