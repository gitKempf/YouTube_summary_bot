"""Telegram Mini App initData HMAC authentication."""
import hashlib
import hmac
import json
import time
from urllib.parse import parse_qs


class TelegramAuthError(Exception):
    pass


def validate_init_data(
    init_data: str, bot_token: str, max_age: int = 300
) -> dict:
    """Validate Telegram initData and return user dict.

    Args:
        init_data: URL-encoded initData string from Telegram WebApp
        bot_token: Bot token for HMAC verification
        max_age: Max seconds since auth_date (default 5 min)

    Returns:
        User dict with id, first_name, username, etc.

    Raises:
        TelegramAuthError: If validation fails
    """
    parsed = parse_qs(init_data)

    received_hash = parsed.get("hash", [None])[0]
    if not received_hash:
        raise TelegramAuthError("Missing hash in initData")

    # Build data-check-string (all params except hash, sorted alphabetically)
    check_pairs = []
    for key in sorted(parsed.keys()):
        if key == "hash":
            continue
        check_pairs.append(f"{key}={parsed[key][0]}")
    data_check_string = "\n".join(check_pairs)

    # HMAC: secret = HMAC_SHA256("WebAppData", bot_token)
    secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    computed_hash = hmac.new(
        secret_key, data_check_string.encode(), hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(computed_hash, received_hash):
        raise TelegramAuthError("Invalid initData signature")

    # Check expiry
    auth_date = int(parsed.get("auth_date", ["0"])[0])
    if time.time() - auth_date > max_age:
        raise TelegramAuthError("initData expired")

    # Parse user
    user_json = parsed.get("user", [None])[0]
    if not user_json:
        raise TelegramAuthError("Missing user in initData")

    return json.loads(user_json)
