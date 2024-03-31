from asr.whisper_live.client import TranscriptionClient

if __name__ == "__main__":
    client = TranscriptionClient(
        "192.168.252.70", 6006
    )
    client("jfk.flac")  # uses microphone audio
