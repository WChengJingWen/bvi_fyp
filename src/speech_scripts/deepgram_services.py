#!/usr/bin/env python3

import rospy
from bvi_fyp.srv import stt, sttResponse
from bvi_fyp.srv import tts, ttsResponse

from deepgram_utils import audio2text, text2audio, get_deepgram_utils

import speech_recognition as sr
from gtts import gTTS
import os
import tempfile
import time


class DeepgramServices:
    def __init__(self):
        rospy.init_node("deepgram_services_node")

        # Ensure utils can init (checks DEEPGRAM_API_KEY etc.)
        utils = get_deepgram_utils()
        if not utils:
            rospy.logerr("DeepgramUtils failed to initialize. Shutting down node.")
            rospy.signal_shutdown("Deepgram init failed")
            return

        # Advertise services
        self.sr_srv = rospy.Service("/speech_to_text", stt, self.handle_sr)
        self.tts_srv = rospy.Service("/text_to_speech", tts, self.handle_tts)

        rospy.loginfo("Deepgram SR (/speech_to_text) and TTS (/text_to_speech) services ready.")

    # Fallback: Google sr
    def google_sr_fallback(self, timeout=5, phrase_time_limit=8):
        """Use SpeechRecognition + Google Web API as a fallback SR."""
        rospy.logwarn("Falling back to Google Speech Recognition...")

        r = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                rospy.loginfo("Google SR: adjusting for ambient noise...")
                r.adjust_for_ambient_noise(source, duration=0.5)

                rospy.loginfo("Google SR: listening...")
                audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

            rospy.loginfo("Google SR: sending to Google...")
            text = r.recognize_google(audio, language='en-US')
            rospy.loginfo(f"Google SR recognized: {text}")
            return text

        except sr.WaitTimeoutError:
            rospy.logwarn("Google SR: no speech detected (timeout).")
        except sr.UnknownValueError:
            rospy.logwarn("Google SR: could not understand audio.")
        except sr.RequestError as e:
            rospy.logerr(f"Google SR: API request error: {e}")
        except Exception as e:
            rospy.logerr(f"Google SR: unexpected error: {e}")

        return ""

    # Fallback: gtts
    def google_tts_fallback(self, text, lang="en"):
        """Use gTTS + system player as a fallback TTS."""
        if not text or not text.strip():
            rospy.logwarn("gTTS fallback: empty text, nothing to say.")
            return False

        rospy.logwarn("Falling back to gTTS for TTS...")
        try:
            tts = gTTS(text=text, lang=lang)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_name = f.name
                tts.write_to_fp(f)

            # Try to play using common CLI players
            played = False
            cmds = [
                f"mpg321 '{tmp_name}' > /dev/null 2>&1",
                f"mplayer '{tmp_name}' > /dev/null 2>&1",
                f"ffplay -nodisp -autoexit '{tmp_name}' > /dev/null 2>&1",
                f"cvlc --play-and-exit '{tmp_name}' > /dev/null 2>&1",
            ]
            for cmd in cmds:
                if os.system(cmd) == 0:
                    played = True
                    break

            if not played:
                rospy.logwarn("gTTS: no audio player (mpg321/mplayer/ffplay/vlc) found in system.")

            # Clean up file
            if os.path.exists(tmp_name):
                os.remove(tmp_name)

            return played
        except Exception as e:
            rospy.logerr(f"gTTS fallback error: {e}")
            return False

    # SR service handler
    def handle_sr(self, req):
        """
        Service handler for speech recognition.
        1) Try Deepgram (via audio2text).
        2) If fail/empty → fallback to Google SR.
        """
        rospy.loginfo("SR service called: listening (Deepgram first)...")

        text = ""
        try:
            text = audio2text(timeout=10, listen_phrase="Please speak now.", use_punctuation_end=False)
        except Exception as e:
            rospy.logerr(f"Deepgram SR raised exception: {e}")
            text = ""

        if not text or not text.strip():
            rospy.logwarn("Deepgram SR failed or returned empty. Using Google SR fallback...")
            text = self.google_sr_fallback(timeout=5, phrase_time_limit=8)

        success = bool(text and text.strip())
        if success:
            rospy.loginfo(f"SR final result: {text}")
        else:
            rospy.logwarn("SR final result: empty (both Deepgram and Google failed).")

        return sttResponse(text=text, success=success)

    # TTS service handler
    def handle_tts(self, req):
        """
        Service handler for text-to-speech.
        1) Try Deepgram (via text2audio).
        2) If fail → fallback to gTTS.
        """
        text = req.text
        rospy.loginfo(f"TTS service called with text: {text}")

        try:
            text2audio(text)   # Deepgram REST via deepgram_utils
            return ttsResponse(success=True)
        except Exception as e:
            rospy.logerr(f"Deepgram TTS error: {e}")
            rospy.logwarn("Using gTTS fallback instead...")

            ok = self.google_tts_fallback(text)
            return ttsResponse(success=ok)


if __name__ == "__main__":
    try:
        node = DeepgramServices()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
