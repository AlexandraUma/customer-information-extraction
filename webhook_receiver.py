from flask import Flask, request, jsonify
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route("/callback", methods=["POST"])
def callback_handler():
    try:
        data = request.json
        logger.info(f"Received callback data: {data}")
        return (
            jsonify({"message": "Callback received successfully!", "data": data}),
            200,
        )
    except Exception as e:
        logger.error(f"Error processing callback: {e}", exc_info=True)
        return jsonify({"message": "Error processing callback", "error": str(e)}), 400


if __name__ == "__main__":
    app.run(port=5000)
