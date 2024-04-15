# python-whisper
A huggingface whisper endpoint for transcribing audio to text in docker.

Step 1. run "python3 download-model.py" in shell

Step 2. run "docker build -t <name> ."

Step 3. run "docker run -p <PORT>:8080 <container>"

Step 4. make POST requests to http://localhost:8080/speech-to-text (with audio bytes as body)