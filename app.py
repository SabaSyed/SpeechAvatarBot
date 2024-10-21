import threading
import subprocess
import av
import pygame
import sys
import vosk
import pyaudio
import json
import time
import ollama
import os

# Paths to idle and talking avatar videos
IDLE_VIDEO = 'idle.mp4'
TALKING_VIDEO = 'speaking.mp4'

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
WINDOW_WIDTH = VIDEO_WIDTH + 20
WINDOW_HEIGHT = VIDEO_HEIGHT + 60

SYSTEM_PROMPT = """You are a friendly and polite voice-based assistant. Please respond concisely and conversationally, as if speaking to the user directly. Avoid technical terms and keep responses simple."""

# Class to manage eSpeak Text-to-Speech functionality
# class TTSManager:
#     def __init__(self):
#         self.speech_completed = threading.Event()

#     def run_espeak_tts(self, text):
#         self.speech_completed.clear()
#         print("TTS (eSpeak) audio started...")
#         try:
#             # Use subprocess to call eSpeak for TTS
#             subprocess.run(['espeak', text], check=True)
#             print("TTS (eSpeak) audio has finished playing.")
#         except subprocess.CalledProcessError as e:
#             print(f"An error occurred while using eSpeak: {e}")
#         self.speech_completed.set()


class TTSManager:
    def __init__(self):
        self.speech_completed = threading.Event()

    def run_espeak_tts(self, text):
        self.speech_completed.clear()
        print("TTS (eSpeak) audio started...")
        try:
            # Check if we need to provide a full path to espeak
            espeak_executable = 'espeak'
            if os.name == 'nt':  # If on Windows
                espeak_executable = r'C:\Program Files\eSpeak\command_line\espeak.exe'

            # Use subprocess to call eSpeak for TTS with options for better clarity
            subprocess.run([espeak_executable, '-s', '150', '-v', 'en', text], check=True)
            print("TTS (eSpeak) audio has finished playing.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while using eSpeak: {e}")
        except FileNotFoundError:
            print("eSpeak executable not found. Please ensure eSpeak is installed and the path is set correctly.")
        self.speech_completed.set()

class VideoManager:
    def __init__(self, screen, video_path):
        self.screen = screen
        self.video_path = video_path
        self.stop_event = threading.Event()

    def play_video(self):
        try:
            container = av.open(self.video_path)
            video_stream = container.streams.video[0]
            frame_rate = float(video_stream.average_rate)
            while not self.stop_event.is_set():
                for frame in container.decode(video=0):
                    img = frame.to_image()
                    frame_surface = pygame.image.frombuffer(img.tobytes(), img.size, img.mode)
                    frame_surface = pygame.transform.scale(frame_surface, (VIDEO_WIDTH, VIDEO_HEIGHT))
                    self.screen.fill((200, 200, 200), (10, 50, VIDEO_WIDTH, VIDEO_HEIGHT))
                    self.screen.blit(frame_surface, (10, 50))
                    pygame.display.flip()
                    pygame.time.delay(int(1000 / (frame_rate * 1.2)))
                    self.handle_ui_events()
                    if self.stop_event.is_set():
                        break
                if not self.stop_event.is_set():
                    container.seek(0)
        except Exception as e:
            print(f"Error playing video: {e}")
        finally:
            container.close()

    def handle_ui_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def stop_video(self):
        self.stop_event.set()

class SpeechManager:

    def __init__(self):
        self.model_path = "vosk-model-small-en-us-0.15"
    
        # self.model_path = "vosk-model-small-en-us-0.15"
        self.model = vosk.Model(self.model_path)
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
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
            else:
                partial_result = self.recognizer.PartialResult()
                partial_text = json.loads(partial_result).get("partial", "")
            if time.time() - silence_start > silence_threshold:
                if full_text.strip():
                    print(f"Final Text: {full_text}")
                    return full_text.strip()
            buffer = b""

    def generate_llama_response(self, prompt):
        try:
            conversation_history = []
            full_prompt = "\n".join([SYSTEM_PROMPT] + conversation_history + [f"User: {prompt}", "Bot:"])
            response = ollama.generate(model="llama3.2:latest", prompt=full_prompt)
            bot_reply = response.get("response", "Sorry, I couldn't generate a response.")
            conversation_history.append(f"User: {prompt}")
            conversation_history.append(f"Bot: {bot_reply}")
            print(bot_reply)
            return bot_reply
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I encountered an issue while generating a response. Could you please try again?"

class AvatarChatbot:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('Avatar Chatbot')
        self.tts_manager = TTSManager()
        self.speech_manager = SpeechManager()

    def run(self):
        stop_idle_event = threading.Event()
        stop_speaking_event = threading.Event()

        idle_video_manager = VideoManager(self.screen, IDLE_VIDEO)
        idle_thread = threading.Thread(target=idle_video_manager.play_video)
        idle_thread.start()

        while True:
            user_input = self.speech_manager.listen()
            if user_input:
                print(f"User said: {user_input}")

                # Generate LLaMA response
                bot_response = self.speech_manager.generate_llama_response(user_input)

                # Stop idle video
                idle_video_manager.stop_video()
                idle_thread.join()

                # Start speaking video and TTS in parallel threads
                speaking_video_manager = VideoManager(self.screen, TALKING_VIDEO)
                speaking_video_thread = threading.Thread(target=speaking_video_manager.play_video)
                tts_thread = threading.Thread(target=self.tts_manager.run_espeak_tts, args=(bot_response,))

                speaking_video_thread.start()
                tts_thread.start()

                # Ensure both TTS and video finish before continuing
                tts_thread.join()
                speaking_video_manager.stop_video()
                speaking_video_thread.join()

                # Restart idle video loop
                idle_video_manager = VideoManager(self.screen, IDLE_VIDEO)
                idle_thread = threading.Thread(target=idle_video_manager.play_video)
                idle_thread.start()

if __name__ == "__main__":
    try:
        avatar_chatbot = AvatarChatbot()
        avatar_chatbot.run()
    except KeyboardInterrupt:
        print("Exiting...")
        pygame.quit()
        sys.exit()
