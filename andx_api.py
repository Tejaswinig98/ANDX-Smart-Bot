"""Minimal, self-contained ANDX client for the beginner bot.

Just the four calls the bot needs, with ANDX's two auth schemes:
  - instant trade (/p/v1/): get_quote + place_instant_order   -> TOKEN-HMAC
  - account     (/api/v1/): get_available + get_order_status   -> API-KEY-HMAC
"""

import hashlib
import hmac
import json
import time

import requests


class APIError(Exception):
    """Raised when ANDX returns status != 'success' (or HTTP 401)."""


class ANDX:
    """ANDX instant-trade + account client (token auth for orders, key auth for balance/status)."""

    BASE = "https://platform.andx.one"

    def __init__(self, user, token, key="", secret="", passphrase="", timeout=15):
        self.user = user
        self.token = token
        self.key = key
        self.secret = secret
        self.passphrase = passphrase
        self.timeout = timeout
        self.session = requests.Session()

    # --- /p/v1/ token auth: quote + order ---

   def _post(self, path, payload):
        """POST a token-signed request; return the data field or raise APIError."""
        body = json.dumps(payload)
        ts = str(int(time.time() * 1000))
        sign = hmac.new(self.token.encode(), (self.user + ts + body).encode(), hashlib.sha256).hexdigest().upper()
        headers = {"access-user": self.user, "access-timestamp": ts, "access-sign": sign,
                   "Content-Type": "application/json"}
        try:
            response = self.session.post(self.BASE + path, data=body, headers=headers, timeout=self.timeout)
            if response.status_code == 401:
                raise APIError("ANDX token expired (401)")
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.RequestException as error:
            raise APIError(f"{path} -> request failed: {error}")
        except ValueError as error:
            raise APIError(f"{path} -> malformed response: {error}")
        if result.get("status") != "success":
            raise APIError(f"{path} -> {result.get('reason')}")
        data = result["data"]
        return data
    def get_quote(self, buy_code, sell_code, sell_amount):
        """Fresh quote for selling `sell_amount` of sell_code into buy_code."""
        quote = self._post("/p/v1/order/get_depth_limits_and_amounts/",
                           {"buy_currency_code": buy_code, "sell_currency_code": sell_code,
                            "sell_currency_amount": sell_amount})
        return quote

    def place_instant_order(self, buy_code, sell_code, sell_amount, buy_amount, visible_price):
        """Execute an instant trade using exact values from a fresh quote."""
        order = self._post("/p/v1/order/instant_order/",
                           {"buy_currency_code": buy_code, "sell_currency_code": sell_code,
                            "sell_amount": sell_amount, "buy_amount": buy_amount,
                            "visible_price": visible_price, "depth_order": True,
                            "with_bonus": False, "with_stake": False})
        return order

    # --- /api/v1/ key auth: balance + order status ---

    def _get(self, path):
        """GET a key-signed request; return the data field or raise APIError."""
        ts = str(time.time())
        msg = self.key + self.user + self.passphrase + ts + "{}"
        sign = hmac.new(self.secret.encode(), msg.encode(), hashlib.sha256).hexdigest().upper()
        headers = {"ACCESS-KEY": self.key, "ACCESS-USER": self.user, "ACCESS-PASSPHRASE": self.passphrase,
                   "ACCESS-TIMESTAMP": ts, "ACCESS-SIGN": sign, "content-type": "application/json"}
        try:
            response = self.session.get(self.BASE + path, headers=headers, timeout=self.timeout)
            if response.status_code == 401:
                raise APIError("ANDX api key rejected (401)")
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.RequestException as error:
            raise APIError(f"{path} -> request failed: {error}")
        except ValueError as error:
            raise APIError(f"{path} -> malformed response: {error}")
        if result.get("status") != "success":
            raise APIError(f"{path} -> {result.get('reason')}")
        data = result["data"]
        return data
    def get_available(self, currency, account="Main"):
        """Spendable balance (float) for one currency."""
        data = self._get(f"/api/v1/balance/{account}/")
        entry = data["balances"].get(currency, {})
        available = float(entry.get("available_balance") or 0)
        return available

    def get_order_status(self, order_number):
        """Order status char: F=filled, C=cancelled, R=rejected, N=new, P=partial."""
        data = self._get(f"/api/v1/order_status/{order_number}/")
        status = data["order"]["status"]
        return status

