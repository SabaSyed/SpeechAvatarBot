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
        self.tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")  # Load once for efficiency

    def run_tts(self, text, reference_audio_path="assets/ref.wav", lang="en"):
        try:
            print("Generating TTS audio...")
            wav_data = self.tts_model.tts(text=text, speaker_wav=reference_audio_path, language=lang)
            audio_array = np.array(wav_data, dtype=np.float32)
            sd.play(audio_array, samplerate=22050)  # Synchronous playback
        except Exception as e:
            print(f"TTS error: {e}")

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
        print("Listening for user input...")

        while True:
            data = self.stream.read(4000, exception_on_overflow=False)
            buffer += data
            if self.recognizer.AcceptWaveform(buffer):
                result = self.recognizer.Result()
                text = json.loads(result).get("text", "")
                if text:
                    print(f"Recognized Text: {text}")
                    full_text += text + " "
                    silence_start = time.time()
            if time.time() - silence_start > silence_threshold and full_text.strip():
                return full_text.strip()
            buffer = b""

    def generate_llama_response(self, prompt):
        try:
            full_prompt = f"{SYSTEM_PROMPT}\nUser: {prompt}\nBot:"
            response = ollama.generate(model="dolphin-llama3", prompt=full_prompt)
            return response.get("response", "I couldn't generate a response.")
        except Exception as e:
            print(f"Llama response generation error: {e}")
            return "Error generating response."

class VideoManager:
    def __init__(self, screen, video_path):
        self.screen = screen
        self.video_path = video_path
        self.stop_event = threading.Event()

    def play_video(self):
        try:
            print(f"Playing video: {self.video_path}")
            container = av.open(self.video_path)
            video_stream = container.streams.video[0]
            frame_rate = float(video_stream.average_rate)

            while not self.stop_event.is_set():
                for frame in container.decode(video=0):
                    if self.stop_event.is_set():
                        break
                    img = frame.to_image()
                    frame_surface = pygame.image.frombuffer(img.tobytes(), img.size, img.mode)
                    
                    # Clear screen and blit video frame centered and scaled
                    self.screen.fill((0, 0, 0))
                    self.screen.blit(pygame.transform.scale(frame_surface, (1024, 1600)), (0, 0))
                    pygame.display.update()
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
        self.screen = pygame.display.set_mode((1024, 1600))  # Set display to match desired video resolution in portrait
        pygame.display.set_caption('Avatar Chatbot')
        self.tts_manager = TTSManager()
        self.speech_manager = SpeechManager()
        self.video_semaphore = threading.Semaphore(1)

    def cleanup(self):
        pygame.quit()
        sys.exit()

    def run(self):
        idle_video_manager = VideoManager(self.screen, IDLE_VIDEO)
        idle_thread = threading.Thread(target=idle_video_manager.play_video)
        idle_thread.start()

        try:
            while True:
                user_input = self.speech_manager.listen()
                if user_input:
                    bot_response = self.speech_manager.generate_llama_response(user_input)

                    # Stop idle video and acquire semaphore for speaking video
                    idle_video_manager.stop_event.set()
                    idle_thread.join()
                    self.video_semaphore.acquire()  # Acquire semaphore before starting speaking video

                    # Play speaking video and TTS response
                    speaking_video_manager = VideoManager(self.screen, TALKING_VIDEO)
                    speaking_thread = threading.Thread(target=speaking_video_manager.play_video)
                    tts_thread = threading.Thread(target=self.tts_manager.run_tts, args=(bot_response,))

                    speaking_thread.start()
                    tts_thread.start()
                    speaking_thread.join()
                    tts_thread.join()

                    # Release semaphore and restart idle video
                    self.video_semaphore.release()
                    idle_video_manager = VideoManager(self.screen, IDLE_VIDEO)
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
