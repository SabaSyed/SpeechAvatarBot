import threading
import queue
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
IDLE_VIDEO = 'idle.mp4'
TALKING_VIDEO = 'speaking.mp4'

SYSTEM_PROMPT = """You are a friendly, chatty, and polite voice-based bot. Please respond concisely and conversationally, as if speaking to the user directly. Avoid technical terms and keep responses simple."""

class TTSManager:
    def __init__(self):
        self.tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")  # Load once for efficiency

    def run_tts(self, text, reference_audio_path="ref.wav", lang="en"):
        try:
            print("Generating TTS audio...")
            wav_data = self.tts_model.tts(text=text, speaker_wav=reference_audio_path, language=lang)
            audio_array = np.array(wav_data, dtype=np.float32)
            sd.play(audio_array, samplerate=22050, blocking=True)
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
        print("Listening for user input...")
        buffer = b""
        full_text = ""
        silence_threshold = 2
        silence_start = time.time()

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
    def __init__(self, screen, video_path, event_queue):
        self.screen = screen
        self.video_path = video_path
        self.event_queue = event_queue

    def play_video(self):
        try:
            container = av.open(self.video_path)
            video_stream = container.streams.video[0]
            frame_rate = max(15.0, float(video_stream.average_rate) - 5)
            while True:
                for frame in container.decode(video=0):
                    img = frame.to_image()
                    frame_surface = pygame.image.frombuffer(img.tobytes(), img.size, img.mode)
                    self.screen.blit(pygame.transform.scale(frame_surface, self.screen.get_size()), (0, 0))
                    pygame.display.flip()
                    pygame.time.delay(int(1000 / frame_rate))
                    self.handle_ui_events()

                    if not self.event_queue.empty():
                        command = self.event_queue.get()
                        if command == "STOP":
                            return
                container.seek(0)  # Loop video if not stopped
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
        display_info = pygame.display.Info()
        screen_width, screen_height = display_info.current_w, display_info.current_h
        self.screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
        pygame.display.set_caption('Avatar Chatbot')

        self.tts_manager = TTSManager()
        self.speech_manager = SpeechManager()
        self.event_queue = queue.Queue()  # Queue for communication between threads
        self.current_video_manager = None  # Keep track of the current video

    def run_pygame_video_loop(self):
        while True:
            if not self.event_queue.empty():
                command = self.event_queue.get()
                if command == "IDLE":
                    self.play_video(IDLE_VIDEO)
                elif command == "TALKING":
                    self.play_video(TALKING_VIDEO)

    def play_video(self, video_path):
        if self.current_video_manager:
            self.event_queue.put("STOP")  # Stop current video
        self.current_video_manager = VideoManager(self.screen, video_path, self.event_queue)
        self.current_video_manager.play_video()

    def main_loop(self):
        pygame_thread = threading.Thread(target=self.run_pygame_video_loop, daemon=True)
        pygame_thread.start()

        try:
            self.event_queue.put("IDLE")  # Start with idle video
            while True:
                user_input = self.speech_manager.listen()
                if user_input:
                    bot_response = self.speech_manager.generate_llama_response(user_input)
                    self.event_queue.put("TALKING")  # Switch to speaking video
                    self.tts_manager.run_tts(bot_response)
                    self.event_queue.put("IDLE")  # Return to idle video after response
        except KeyboardInterrupt:
            print("Exiting...")
            self.cleanup()

    def cleanup(self):
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    avatar_chatbot = AvatarChatbot()
    avatar_chatbot.main_loop()
