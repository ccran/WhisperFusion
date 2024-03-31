from asr.whisper_live.faster_whisper_serve_client import TranscriptionServer

if __name__ == "__main__":
    server = TranscriptionServer()
    server.run("0.0.0.0", 6006)