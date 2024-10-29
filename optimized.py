import threading
import av
import pygame
import sys
import vosk
import json
import time
import ollama
import pyaudio
import numpy as np
from TTS.api import TTS
import sounddevice as sd

# Paths to idle and talking avatar videos
IDLE_VIDEO = 'assets/idle.mp4'
TALKING_VIDEO = 'assets/speaking.mp4'

SYSTEM_PROMPT = """You are a friendly, chatty, and polite voice-based bot. Please respond concisely and conversationally, as if speaking to the user directly. Avoid technical terms and keep responses simple."""

class TTSManager:
    def __init__(self):
        self.speech_completed = threading.Event()
        self.tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

    def run_tts(self, text, reference_audio_path="assets/ref.wav", lang="en"):
        self.speech_completed.clear()
        try:
            wav_data = self.tts_model.tts(text=text, speaker_wav=reference_audio_path, language=lang)
            audio_array = np.array(wav_data, dtype=np.float32)
            sd.play(audio_array, samplerate=22050, blocking=True)
        except Exception as e:
            print(f"TTS error: {e}")
        finally:
            self.speech_completed.set()

class SpeechManager:
    def __init__(self):
        self.model = vosk.Model("vosk-model-small-en-us-0.15")
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=16000)
        self.stream.start_stream()

    def listen(self):
        buffer = b""
        full_text = ""
        silence_threshold = 2
        silence_start = time.time()

        while True:
            try:
                data = self.stream.read(4000, exception_on_overflow=False)
                buffer += data
                if self.recognizer.AcceptWaveform(buffer):
                    result = self.recognizer.Result()
                    text = json.loads(result).get("text", "")
                    if text:
                        full_text += text + " "
                        silence_start = time.time()
            except Exception as e:
                print(f"Audio stream error: {e}")
                continue
            if time.time() - silence_start > silence_threshold and full_text.strip():
                return full_text.strip()
            buffer = b""

    def generate_llama_response(self, prompt):
        try:
            full_prompt = f"{SYSTEM_PROMPT}\nUser: {prompt}\nBot:"
            response = ollama.generate(model="dolphin-llama3", prompt=full_prompt)
            return response.get("response", "I couldn't generate a response.")
        except Exception as e:
            return "Error generating response."

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

class VideoManager:
    def __init__(self, screen, video_path, semaphore):
        self.screen = screen
        self.video_path = video_path
        self.semaphore = semaphore

    def play_video(self):
        try:
            container = av.open(self.video_path)
            video_stream = container.streams.video[0]
            frame_rate = max(15.0, float(video_stream.average_rate) - 5)

            with self.semaphore:
                for frame in container.decode(video=0):
                    img = frame.to_image()
                    frame_surface = pygame.image.frombuffer(img.tobytes(), img.size, img.mode)
                    self.screen.blit(pygame.transform.scale(frame_surface, self.screen.get_size()), (0, 0))
                    pygame.display.flip()
                    pygame.time.delay(int(1000 / frame_rate))
                    self.handle_ui_events()
                container.close()
        except Exception as e:
            print(f"Error playing video: {e}")

    def handle_ui_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

class AvatarChatbot:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((640, 480))
        pygame.display.set_caption('Avatar Chatbot')
        self.tts_manager = TTSManager()
        self.speech_manager = SpeechManager()
        self.video_semaphore = threading.Semaphore(1)

    def cleanup(self):
        self.speech_manager.close()
        pygame.quit()
        sys.exit()

    def run(self):
        idle_video_manager = VideoManager(self.screen, IDLE_VIDEO, self.video_semaphore)
        idle_thread = threading.Thread(target=idle_video_manager.play_video)
        idle_thread.start()

        try:
            while True:
                user_input = self.speech_manager.listen()
                if user_input:
                    bot_response = self.speech_manager.generate_llama_response(user_input)

                    # Stop the idle video
                    self.video_semaphore.acquire()
                    speaking_video_manager = VideoManager(self.screen, TALKING_VIDEO, self.video_semaphore)
                    speaking_thread = threading.Thread(target=speaking_video_manager.play_video)
                    speaking_thread.start()

                    # Run TTS in non-blocking mode
                    tts_thread = threading.Thread(target=self.tts_manager.run_tts, args=(bot_response,))
                    tts_thread.start()

                    tts_thread.join()
                    speaking_thread.join()
                    self.video_semaphore.release()

                    # Re-launch idle video after response completes
                    idle_video_manager = VideoManager(self.screen, IDLE_VIDEO, self.video_semaphore)
                    idle_thread = threading.Thread(target=idle_video_manager.play_video)
                    idle_thread.start()

        except Exception as e:
            print(f"Error occurred: {e}")
            self.cleanup()


if __name__ == "__main__":
    try:
        avatar_chatbot = AvatarChatbot()
        avatar_chatbot.run()
    except KeyboardInterrupt:
        print("Exiting...")
        avatar_chatbot.cleanup()
