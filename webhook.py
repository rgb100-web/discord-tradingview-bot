# webhook.py
from flask import Flask, request, jsonify
import logging

log = logging.getLogger("webhook")

def create_app(ai_engine):
    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def health():
        return "OK", 200

    @app.route("/webhook", methods=["POST"])
    def webhook():
        """
        Expected JSON from TradingView (example format below).
        This handler simply forwards payload to the AIEngine for evaluation.
        """
        try:
            payload = request.get_json(force=True)
            # Basic validation / sanitization
            symbol = payload.get("symbol") or payload.get("ticker") or payload.get("instrument")
            if not symbol:
                return jsonify({"error":"missing symbol"}), 400

            # Run AI evaluation async (do not block request)
            ai_engine.enqueue(payload)
            return jsonify({"status":"queued"}), 202
        except Exception as e:
            log.exception("Webhook error")
            return jsonify({"error": str(e)}), 500

    return app
