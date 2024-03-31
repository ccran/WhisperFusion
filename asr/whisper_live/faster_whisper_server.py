from asr.whisper_live.vad import VoiceActivityDetection
from websockets.sync.server import serve
import torch
import functools
import json
import numpy as np
import time
from asr.whisper_live.transcriber import WhisperModel
from asr.whisper_live.faster_whisper_serve_client import ServeClient

import logging

logging.basicConfig(format='%(asctime)s %(filename)s %(lineno)d %(levelname)s %(message)s',
                    datefmt='%a %d %b %Y %H:%M:%S', level=logging.INFO)


class FasterWhisperServer:
    """
    Represents a transcription server that handles incoming audio from clients.

    Attributes:
        RATE (int): The audio sampling rate (constant) set to 16000.
        vad_model (torch.Module): The voice activity detection model.
        vad_threshold (float): The voice activity detection threshold.
        clients (dict): A dictionary to store connected clients.
        websockets (dict): A dictionary to store WebSocket connections.
        clients_start_time (dict): A dictionary to track client start times.
        max_clients (int): Maximum allowed connected clients.
        max_connection_time (int): Maximum allowed connection time in seconds.
    """

    RATE = 16000

    def __init__(self):
        # voice activity detection model

        self.clients = {}
        self.websockets = {}
        self.clients_start_time = {}
        self.max_clients = 4
        self.max_connection_time = 600
        self.transcriber = None
        self.vad_model = None
        self.vad_threshold = 0.5
        self.no_voice_activity_chunks = 0

    # 获取客户端链接需要等待的时间
    def get_wait_time(self):
        """
        Calculate and return the estimated wait time for clients.

        Returns:
            float: The estimated wait time in minutes.
        """
        wait_time = None

        for k, v in self.clients_start_time.items():
            current_client_time_remaining = self.max_connection_time - (time.time() - v)

            if wait_time is None or current_client_time_remaining < wait_time:
                wait_time = current_client_time_remaining

        return wait_time / 60

    # 最大连接时间判断
    def exceed_max_connection_time(self, websocket):
        elapsed_time = time.time() - self.clients_start_time[websocket]
        if elapsed_time >= self.max_connection_time:
            self.clients[websocket].disconnect()
            logging.warning(f"{self.clients[websocket]} Client disconnected due to overtime.")
            self.clients[websocket].cleanup()
            self.clients.pop(websocket)
            self.clients_start_time.pop(websocket)
            websocket.close()
            del websocket
            return True
        return False

    # 清理websocket
    def clean_websocket(self, websocket):
        self.clients[websocket].cleanup()
        self.clients.pop(websocket)
        self.clients_start_time.pop(websocket)
        logging.info("Connection Closed.")
        logging.info(self.clients)
        del websocket
        return

    # 最大连接数量判断
    def exceed_max_connection_num(self, websocket, uid):
        if len(self.clients) >= self.max_clients:
            logging.warning("Client Queue Full. Asking client to wait ...")
            wait_time = self.get_wait_time()
            response = {
                "uid": uid,
                "status": "WAIT",
                "message": wait_time,
            }
            websocket.send(json.dumps(response))
            websocket.close()
            del websocket
            return True
        return False

    # vad检测
    def vad_check(self, websocket, frame_np):
        # VAD
        try:
            speech_prob = self.vad_model(torch.from_numpy(frame_np.copy()), self.RATE).item()
            if speech_prob < self.vad_threshold:
                self.no_voice_activity_chunks += 1
                if self.no_voice_activity_chunks > 3:
                    if not self.clients[websocket].eos:
                        self.clients[websocket].set_eos(True)
                    time.sleep(0.1)  # EOS stop receiving frames for a 100ms(to send output to LLM.)
                return False
            self.no_voice_activity_chunks = 0
            self.clients[websocket].set_eos(False)
        except Exception as e:
            logging.error(e)
            return False
        return True

    def recv_audio(self, websocket,
                   transcription_queue=None,
                   llm_queue=None,
                   vad_path=None,
                   faster_whisper_path=None,
                   should_send_server_ready=None):
        """
        Receive audio chunks from a client in an infinite loop.

        Continuously receives audio frames from a connected client
        over a WebSocket connection. It processes the audio frames using a
        voice activity detection (VAD) model to determine if they contain speech
        or not. If the audio frame contains speech, it is added to the client's
        audio data for ASR.
        If the maximum number of clients is reached, the method sends a
        "WAIT" status to the client, indicating that they should wait
        until a slot is available.
        If a client's connection exceeds the maximum allowed time, it will
        be disconnected, and the client's resources will be cleaned up.

        Args:
            websocket (WebSocket): The WebSocket connection for the client.

        Raises:
            Exception: If there is an error during the audio frame processing.
        """

        # 接收客户端连接信息
        logging.info("New client connected")
        options = websocket.recv()
        options = json.loads(options)
        logging.info("received options: {}".format(options))

        # 连接数判断
        if self.exceed_max_connection_num(websocket, options["uid"]):
            return

        # 创建vad模型
        if self.vad_model is None:
            self.vad_model = VoiceActivityDetection(path=vad_path)
        # 创建whisper
        if self.transcriber is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.transcriber = WhisperModel(
                faster_whisper_path,
                # "small" if multilingual else "small.en",
                device=device,
                compute_type="int8" if device == "cpu" else "float16",
                local_files_only=False,
            )

        # 启动websocket服务线程
        client = ServeClient(
            websocket,
            multilingual=options["multilingual"],
            language=options["language"],
            task=options["task"],
            client_uid=options["uid"],
            transcription_queue=transcription_queue,
            llm_queue=llm_queue,
            transcriber=self.transcriber,
        )

        self.clients[websocket] = client
        self.clients_start_time[websocket] = time.time()
        while True:
            try:
                # 添加音频帧处理
                frame_data = websocket.recv()
                frame_np = np.frombuffer(frame_data, dtype=np.float32)
                # vad检测
                is_vad = self.vad_check(websocket, frame_np)
                if is_vad:
                    self.clients[websocket].add_frames(frame_np)
                # 最大连接时间判断
                if self.exceed_max_connection_time(websocket):
                    break
            except Exception as e:
                logging.error(e)
                self.clean_websocket(websocket)
                break

    def run(self, host, port=9090,
            transcription_queue=None,
            llm_queue=None,
            vad_path=None,
            faster_whisper_path=None,
            should_send_server_ready=None):
        """
        Run the transcription server.

        Args:
            host (str): The host address to bind the server.
            port (int): The port number to bind the server.
        """
        # wait for WhisperSpeech to warmup
        while not should_send_server_ready.value:
            time.sleep(0.5)

        with serve(
                functools.partial(
                    self.recv_audio,
                    vad_path=vad_path,
                    transcription_queue=transcription_queue,
                    llm_queue=llm_queue,
                    faster_whisper_path=faster_whisper_path,
                    should_send_server_ready=should_send_server_ready,
                ),
                host,
                port
        ) as server:
            logging.info("start server...")
            server.serve_forever()
