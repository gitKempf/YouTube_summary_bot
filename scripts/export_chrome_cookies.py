#!/usr/bin/env python3
"""Export YouTube cookies from Chrome browser to cookies.txt (Netscape format).

Works on macOS by reading Chrome's SQLite cookie DB and decrypting
with the key from macOS Keychain.

Usage:
    python scripts/export_chrome_cookies.py
    # Outputs cookies.txt in project root
"""
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from hashlib import pbkdf2_hmac
from pathlib import Path

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
except ImportError:
    print("Install cryptography: pip install cryptography")
    sys.exit(1)

# Chrome cookie DB path (macOS)
CHROME_COOKIE_DB = Path.home() / "Library/Application Support/Google/Chrome/Default/Cookies"
CHROME_PROFILES = [
    Path.home() / "Library/Application Support/Google/Chrome/Default/Cookies",
    Path.home() / "Library/Application Support/Google/Chrome/Profile 1/Cookies",
    Path.home() / "Library/Application Support/Google/Chrome/Profile 2/Cookies",
]


def get_chrome_key_macos() -> bytes:
    """Get Chrome's encryption key from macOS Keychain."""
    proc = subprocess.run(
        ["security", "find-generic-password", "-s", "Chrome Safe Storage", "-w"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Could not read Chrome Safe Storage key from Keychain.\n"
            "Make sure Chrome is installed and you've logged in at least once."
        )
    password = proc.stdout.strip()
    # Chrome uses PBKDF2 with 1003 iterations, 16-byte key
    return pbkdf2_hmac("sha1", password.encode("utf-8"), b"saltysalt", 1003, dklen=16)


def decrypt_cookie(encrypted_value: bytes, key: bytes) -> str:
    """Decrypt a Chrome cookie value."""
    if not encrypted_value:
        return ""

    # v10/v11 prefix (3 bytes)
    if encrypted_value[:3] in (b"v10", b"v11"):
        encrypted_value = encrypted_value[3:]
    else:
        # Not encrypted
        return encrypted_value.decode("utf-8", errors="replace")

    # AES-128-CBC, IV is 16 bytes of spaces
    iv = b" " * 16
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    decrypted = decryptor.update(encrypted_value) + decryptor.finalize()

    # Remove PKCS7 padding
    padding_len = decrypted[-1]
    if padding_len < 16:
        decrypted = decrypted[:-padding_len]

    return decrypted.decode("utf-8", errors="replace")


def chrome_timestamp_to_unix(chrome_ts: int) -> int:
    """Convert Chrome's microsecond epoch (Jan 1, 1601) to Unix epoch."""
    if chrome_ts == 0:
        return 0
    # Chrome epoch offset: seconds between 1601-01-01 and 1970-01-01
    return int((chrome_ts / 1_000_000) - 11644473600)


def find_cookie_db() -> Path:
    """Find the Chrome cookie database file."""
    for path in CHROME_PROFILES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Chrome cookie database not found. Checked:\n" +
        "\n".join(f"  {p}" for p in CHROME_PROFILES)
    )


def export_cookies(output_path: str = "cookies.txt", domain: str = ".youtube.com"):
    """Export cookies for a domain from Chrome to Netscape cookies.txt format."""
    db_path = find_cookie_db()
    print(f"Found Chrome cookies at: {db_path}")

    # Copy DB to temp (Chrome locks it)
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = tmp.name
    shutil.copy2(db_path, tmp_path)

    try:
        key = get_chrome_key_macos()
        print("Got encryption key from Keychain")

        conn = sqlite3.connect(tmp_path)
        cursor = conn.cursor()

        # Query cookies for youtube.com and google.com (auth cookies)
        domains = [".youtube.com", ".google.com", "youtube.com", "www.youtube.com"]
        placeholders = ",".join("?" * len(domains))
        cursor.execute(
            f"SELECT host_key, name, encrypted_value, path, expires_utc, is_secure, is_httponly "
            f"FROM cookies WHERE host_key IN ({placeholders})",
            domains,
        )

        rows = cursor.fetchall()
        conn.close()
        print(f"Found {len(rows)} cookies for YouTube/Google domains")

        if not rows:
            print("\nNo cookies found. Make sure you're logged into YouTube in Chrome.")
            sys.exit(1)

        # Write Netscape cookies.txt format
        lines = ["# Netscape HTTP Cookie File", "# Exported by export_chrome_cookies.py", ""]
        for host, name, enc_val, path, expires, secure, httponly in rows:
            value = decrypt_cookie(enc_val, key)
            exp_unix = chrome_timestamp_to_unix(expires)
            is_domain = "TRUE" if host.startswith(".") else "FALSE"
            is_secure = "TRUE" if secure else "FALSE"
            lines.append(f"{host}\t{is_domain}\t{path}\t{is_secure}\t{exp_unix}\t{name}\t{value}")

        Path(output_path).write_text("\n".join(lines) + "\n")
        print(f"\nExported {len(rows)} cookies to {output_path}")
        print(f"Upload to server: scp {output_path} contabo:/opt/youtube-summary-bot/cookies.txt")

    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "cookies.txt"
    export_cookies(output)
