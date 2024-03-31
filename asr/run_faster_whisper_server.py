import sys
import os

sys.path.append("/home/ccran/WhisperFusion")
from asr.whisper_live.faster_whisper_server import FasterWhisperServer
from multiprocessing import Value, Queue
import ctypes

if __name__ == "__main__":
    transcription_queue = Queue()
    llm_queue = Queue()
    should_send_server_ready = Value(ctypes.c_bool, True)
    vad_path = '/home/ccran/vad/silero_vad.onnx'
    faster_whisper_path = '/home/ccran/fasterwhisper/faster-whisper-large-v3'
    server = FasterWhisperServer()
    server.run("0.0.0.0", 6006,
               transcription_queue,
               llm_queue,
               vad_path,
               faster_whisper_path,
               should_send_server_ready)
