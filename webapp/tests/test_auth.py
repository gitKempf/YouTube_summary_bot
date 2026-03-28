"""Tests for Telegram initData HMAC authentication."""
import hashlib
import hmac
import time
import pytest
from urllib.parse import urlencode

from webapp.auth import validate_init_data, TelegramAuthError


def _make_init_data(bot_token: str, user_id: int = 42, override_hash: str = None, auth_date: int = None):
    """Create valid Telegram initData string with proper HMAC."""
    if auth_date is None:
        auth_date = int(time.time())
    data = {
        "user": f'{{"id":{user_id},"first_name":"Test","username":"testuser"}}',
        "auth_date": str(auth_date),
    }
    # Build data-check-string (sorted key=value pairs, excluding hash)
    data_check_string = "\n".join(f"{k}={v}" for k, v in sorted(data.items()))
    # HMAC: secret = HMAC_SHA256("WebAppData", bot_token)
    secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    computed_hash = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()

    data["hash"] = override_hash or computed_hash
    return urlencode(data)


class TestTelegramAuth:
    def test_validates_correct_initdata(self):
        token = "123456:ABC"
        init_data = _make_init_data(token)
        user = validate_init_data(init_data, token)
        assert user["id"] == 42
        assert user["first_name"] == "Test"

    def test_rejects_invalid_hash(self):
        token = "123456:ABC"
        init_data = _make_init_data(token, override_hash="badhash123")
        with pytest.raises(TelegramAuthError, match="Invalid"):
            validate_init_data(init_data, token)

    def test_rejects_expired(self):
        token = "123456:ABC"
        old_time = int(time.time()) - 600  # 10 minutes ago
        init_data = _make_init_data(token, auth_date=old_time)
        with pytest.raises(TelegramAuthError, match="expired"):
            validate_init_data(init_data, token, max_age=300)

    def test_extracts_user_id(self):
        token = "123456:ABC"
        init_data = _make_init_data(token, user_id=99999)
        user = validate_init_data(init_data, token)
        assert user["id"] == 99999
