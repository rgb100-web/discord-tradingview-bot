# webhook.py
from flask import Flask, request, jsonify
import logging

log = logging.getLogger("webhook")

def create_app(ai_engine):
    app = Flask(__name__)

    @app.route("/webhook", methods=["POST"])
    def webhook():
        try:
            payload = request.get_json(force=True)
        except Exception as e:
            log.error("Webhook JSON parse error")
            log.exception(e)
            return jsonify({"error": "Invalid JSON format"}), 400

        if not isinstance(payload, dict):
            return jsonify({"error": "Payload must be JSON object"}), 400

        ai_engine.enqueue(payload)
        return jsonify({"status": "queued"}), 202

    return app
