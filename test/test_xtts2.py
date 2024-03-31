from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

config = XttsConfig()
config.load_json("/home/ccran/xtts/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="/home/ccran/xtts", eval=True)
model.cuda()

out = model.inference(
    text="It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    language="en",
    gpt_cond_latent=gpt_cond_latent,
    speaker_embedding=speaker_embedding,
    temperature=XTTS_MODEL.config.temperature,  # Add custom parameters here
    length_penalty=XTTS_MODEL.config.length_penalty,
    repetition_penalty=XTTS_MODEL.config.repetition_penalty,
    top_k=XTTS_MODEL.config.top_k,
    top_p=XTTS_MODEL.config.top_p,
)
# outputs = model.synthesize(
#     "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
#     config,
#     speaker_wav="example.wav",
#     gpt_cond_len=3,
#     language="en",
# )
