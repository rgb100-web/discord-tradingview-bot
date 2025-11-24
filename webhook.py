# webhook.py
import logging
from flask import Flask, request, jsonify

log = logging.getLogger("webhook")


def create_app(ai_engine):
    app = Flask(__name__)

    @app.route("/webhook", methods=["POST"])
    def webhook():
        try:
            payload = request.get_json(force=True)
            if not isinstance(payload, dict):
                raise ValueError("Payload must be a JSON object")

            # Hand off to AI engine queue
            ai_engine.enqueue_alert(payload)
            log.info(f"Webhook received and queued: {payload}")

            return jsonify({"status": "queued"}), 202

        except Exception as e:
            log.exception("Webhook error")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/", methods=["GET"])
    def root():
        return "Discord TradingView Bot is running", 200

    return app
