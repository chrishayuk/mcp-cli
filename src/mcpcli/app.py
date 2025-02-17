from flask import Flask, render_template, request, jsonify
import subprocess
import os

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

@app.route("/")
def home():
    return render_template("index.html")  # Serves the frontend UI

@app.route("/run", methods=["POST"])
def run_command():
    data = request.json
    command = data.get("command")

    if command == "start":
        try:
            # Use the correct path to llm_client.py
            result = subprocess.run(
                ["uv", "run", "mcp-cli", "--server", "sqlite", "--server", "fetch", "--provider", "groq", "--model", "mixtral-8x7b-32768"],
                capture_output=True,
                text=True,
                check=True
            )
            return jsonify({"output": result.stdout})
        except subprocess.CalledProcessError as e:
            return jsonify({"error": f"Error running llm_client.py: {e}"})

    elif command == "config":
        return jsonify({"output": "Opening configuration settings..."})

    elif command == "help":
        return jsonify({"output": "Available commands: start, config, help, exit"})

    elif command == "exit":
        return jsonify({"output": "Goodbye!"})

    else:
        return jsonify({"output": "Invalid command!"})


if __name__ == "__main__":
    app.run(debug=True)
