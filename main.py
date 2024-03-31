import multiprocessing
import argparse
import ctypes

from multiprocessing import Manager, Value, Queue

from asr.whisper_live.faster_whisper_server import FasterWhisperServer
from llm.llm_service import TensorRTLLMEngine
from tts.tts_service import WhisperSpeechTTS


def parse_arguments():
    parser = argparse.ArgumentParser()
    # vad路径
    parser.add_argument('--vad_path',
                        type=str,
                        default="/home/ccran/vad",
                        help='vad model path')
    # faster-whisper路径
    parser.add_argument('--faster_whisper_path',
                        type=str,
                        default="/home/ccran/fasterwhisper/faster-whisper-large-v3",
                        help='faster whisper model path')
    return parser.parse_args()


if __name__ == "__main__":
    # 1、接收参数
    args = parse_arguments()
    if not args.faster_whisper_path:
        raise ValueError("Please provide faster_whisper_path to run the pipeline.")
        import sys

        sys.exit(0)

    # 2、启动多进程 创建锁和队列
    multiprocessing.set_start_method('spawn')
    lock = multiprocessing.Lock()
    manager = Manager()
    shared_output = manager.list()
    should_send_server_ready = Value(ctypes.c_bool, True)
    transcription_queue = Queue()
    llm_queue = Queue()
    audio_queue = Queue()

    # 3、启动asr
    whisper_server = FasterWhisperServer()
    whisper_process = multiprocessing.Process(
        target=whisper_server.run,
        args=(
            "0.0.0.0",
            6006,
            transcription_queue,
            llm_queue,
            args.vad_path,
            args.faster_whisper_path,
            should_send_server_ready
        )
    )
    whisper_process.start()

    # 4、启动llm
    # llm_provider = TensorRTLLMEngine()
    # # llm_provider = MistralTensorRTLLMProvider()
    # llm_process = multiprocessing.Process(
    #     target=llm_provider.run,
    #     args=(
    #         # args.mistral_tensorrt_path,
    #         # args.mistral_tokenizer_path,
    #         args.phi_tensorrt_path,
    #         args.phi_tokenizer_path,
    #         transcription_queue,
    #         llm_queue,
    #         audio_queue,
    #     )
    # )
    # llm_process.start()

    # 5、启动tts
    # tts_runner = WhisperSpeechTTS()
    # tts_process = multiprocessing.Process(target=tts_runner.run,
    #                                       args=("0.0.0.0", 8888, audio_queue, should_send_server_ready))
    # tts_process.start()

    # 6、等待所有子进程结束
    whisper_process.join()
    # llm_process.join()
    # tts_process.join()
