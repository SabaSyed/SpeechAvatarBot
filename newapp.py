import threading
import av
import pygame
import sys
import vosk
import json
import time
import ollama
import os
import numpy as np
from TTS.api import TTS
import sounddevice as sd

# Paths to idle and talking avatar videos
IDLE_VIDEO = 'assets/idle.mp4'
TALKING_VIDEO = 'assets/speaking.mp4'

SYSTEM_PROMPT = """You are a friendly, chatty and polite voice-based bot. Please respond concisely and conversationally, as if speaking to the user directly. Avoid technical terms and keep responses simple."""

class TTSManager:
    def __init__(self):
        self.speech_completed = threading.Event()
        self.tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

    def run_tts(self, text, reference_audio_path="assets/ref.wav", lang="en"):
        self.speech_completed.clear()
        print("TTS (Coqui TTS) audio generation started...")

        try:
            wav_data = self.tts_model.tts(text=text, speaker_wav=reference_audio_path, language=lang)
            audio_array = np.array(wav_data, dtype=np.float32)
            print("TTS (Coqui TTS) audio generated, now playing...")
            sd.play(audio_array, samplerate=22050)
            sd.wait()
        except Exception as e:
            print(f"An error occurred while generating TTS audio: {e}")

        self.speech_completed.set()

class SpeechManager:
    def __init__(self):
        self.model_path = "vosk-model-small-en-us-0.15"
        self.model = vosk.Model(self.model_path)
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000)

    def listen(self):
        print("Listening for user input...")
        full_text = ""
        silence_threshold = 2
        silence_start = time.time()

        def callback(indata, frames, time_info, status):
            if status:
                print(status)
            if self.recognizer.AcceptWaveform(indata.tobytes()):
                result = self.recognizer.Result()
                text = json.loads(result).get("text", "")
                if text:
                    print(f"Recognized Text: {text}")
                    full_text += text + " "
                    silence_start = time.time()

        with sd.InputStream(callback=callback, channels=1, samplerate=16000):
            while True:
                if time.time() - silence_start > silence_threshold:
                    if full_text.strip():
                        print(f"Final Text: {full_text}")
                        return full_text.strip()

    def generate_llama_response(self, prompt):
        try:
            full_prompt = "\n".join([SYSTEM_PROMPT] + [f"User: {prompt}", "Bot:"])
            response = ollama.generate(model="dolphin-llama3", prompt=full_prompt)
            bot_reply = response.get("response", "Sorry, I couldn't generate a response.")
            print(bot_reply)
            return bot_reply
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I encountered an issue while generating a response. Could you please try again."

class VideoManager:
    def __init__(self, screen, video_path, stop_event):
        self.screen = screen
        self.video_path = video_path
        self.stop_event = stop_event

    def play_video(self):
        try:
            container = av.open(self.video_path)
            video_stream = container.streams.video[0]
            frame_rate = float(video_stream.average_rate)
            while not self.stop_event.is_set():
                for frame in container.decode(video=0):
                    img = frame.to_image()
                    frame_surface = pygame.image.frombuffer(img.tobytes(), img.size, img.mode)
                    frame_surface = pygame.transform.scale(frame_surface, (self.screen.get_width(), self.screen.get_height()))
                    self.screen.fill((200, 200, 200))
                    self.screen.blit(frame_surface, (0, 0))
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
                self.stop_event.set()
                pygame.quit()
                sys.exit()

class AvatarChatbot:
    def __init__(self):
        pygame.init()
        self.info = pygame.display.Info()
        self.screen = pygame.display.set_mode((self.info.current_w, self.info.current_h))
        pygame.display.set_caption('Avatar Chatbot')
        self.tts_manager = TTSManager()
        self.speech_manager = SpeechManager()
        self.stop_event = threading.Event()

    def cleanup(self):
        print("Cleaning up resources...")
        pygame.quit()
        sys.exit()

    def run(self):
        idle_video_manager = VideoManager(self.screen, IDLE_VIDEO, self.stop_event)
        idle_thread = threading.Thread(target=idle_video_manager.play_video)
        idle_thread.start()

        try:
            while not self.stop_event.is_set():
                try:
                    user_input = self.speech_manager.listen()
                except Exception as e:
                    print(f"Error in voice recognition: {e}")
                    continue

                if user_input:
                    print(f"User said: {user_input}")
                    bot_response = self.speech_manager.generate_llama_response(user_input)

                    idle_video_manager.stop_event.set()
                    idle_thread.join()

                    speaking_video_manager = VideoManager(self.screen, TALKING_VIDEO, self.stop_event)
                    speaking_video_thread = threading.Thread(target=speaking_video_manager.play_video)
                    tts_thread = threading.Thread(target=self.tts_manager.run_tts, args=(bot_response,))

                    speaking_video_thread.start()
                    tts_thread.start()

                    tts_thread.join()
                    speaking_video_manager.stop_event.set()
                    speaking_video_thread.join()

                    # Reset stop event for next loop
                    self.stop_event.clear()
                    idle_video_manager = VideoManager(self.screen, IDLE_VIDEO, self.stop_event)
                    idle_thread = threading.Thread(target=idle_video_manager.play_video)
                    idle_thread.start()

        except Exception as e:
            print(f"Error occurred: {e}")
            self.stop_event.set()
        finally:
            self.cleanup()


if __name__ == "__main__":
    try:
        avatar_chatbot = AvatarChatbot()
        avatar_chatbot.run()
    except KeyboardInterrupt:
        print("Exiting...")
        avatar_chatbot.stop_event.set()
        avatar_chatbot.cleanup()
