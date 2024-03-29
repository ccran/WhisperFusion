from whisper_live.transcriber import WhisperModel
#from faster_whisper import WhisperModel
model_size = "/home/ccran/fasterwhisper/faster-whisper-large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("/home/ccran/fasterwhisper/faster-whisper-large-v3/example.mp3",language='zh',beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
