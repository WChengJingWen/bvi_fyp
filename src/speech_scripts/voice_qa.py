#!/usr/bin/env python3

import rospy
from bvi_fyp.srv import stt, sttRequest
from bvi_fyp.srv import tts, ttsRequest
from bvi_fyp.srv import rag, ragRequest

class VoiceQANode:
    def __init__(self):
        rospy.init_node("voice_rag_node")

        # Keep a short chat history in memory
        self.history_user = []
        self.history_bot = []

        # Wait for services
        rospy.loginfo("Waiting for /speech_to_text, /text_to_speech, /rag_query services...")
        rospy.wait_for_service("/speech_to_text")
        rospy.wait_for_service("/text_to_speech")
        rospy.wait_for_service("/rag_query")

        self.sr_client = rospy.ServiceProxy("/speech_to_text", stt)
        self.tts_client = rospy.ServiceProxy("/text_to_speech", tts)
        self.rag_client = rospy.ServiceProxy("/rag_query", rag)

        rospy.loginfo("Voice RAG node started. Ready to chat.")
        self.main_loop()

    def speak(self, text):
        try:
            self.tts_client(ttsRequest(text=text))
        except Exception as e:
            rospy.logerr(f"TTS call failed: {e}")

    def listen(self):
        try:
            resp = self.sr_client(sttRequest())
            if resp.success and resp.text.strip():
                return resp.text.strip()
            return None
        except Exception as e:
            rospy.logerr(f"SR call failed: {e}")
            return None

    def ask_rag(self, query):
        req = ragRequest()
        req.query = query
        req.history_user = self.history_user
        req.history_bot = self.history_bot

        resp = self.rag_client(req)
        return resp.answer

    def main_loop(self):
        # simple loop: listen → answer → speak
        self.speak("Hello, I am your campus guide robot. You can ask me about locations on campus.")

        rate = rospy.Rate(0.1)  # e.g. one iteration every 10 seconds, or replace with event triggers
        while not rospy.is_shutdown():
            self.speak("Please ask your question, or say 'exit' to stop.")
            user_text = self.listen()
            if not user_text:
                self.speak("Sorry, I did not hear anything.")
                continue

            rospy.loginfo(f"User said: {user_text}")

            if user_text.lower() in ["exit", "quit", "stop"]:
                self.speak("Goodbye.")
                break

            # Call RAG
            answer = self.ask_rag(user_text)
            rospy.loginfo(f"RAG answer: {answer}")

            # Update history (limit length)
            self.history_user.append(user_text)
            self.history_bot.append(answer)
            if len(self.history_user) > 5:
                self.history_user = self.history_user[-5:]
                self.history_bot = self.history_bot[-5:]

            # Speak answer
            self.speak(answer)

            # example: you can also add "Do you need me to guide you there?" here conditionally

            rate.sleep()

    # def main_loop(self):
    #     print("🧪 TEXT MODE: RAG testing without speech")
    #     print("Type your campus question below (type 'exit' to quit):\n")

    #     while not rospy.is_shutdown():
    #         user_text = input("You: ").strip()
    #         if user_text.lower() in ["exit", "quit", "stop"]:
    #             print("Goodbye.")
    #             break

    #         # Call RAG service
    #         answer = self.ask_rag(user_text)
    #         print("\n🤖 Robot:", answer, "\n")

    #         # Update chat history
    #         self.history_user.append(user_text)
    #         self.history_bot.append(answer)
    #         if len(self.history_user) > 5:
    #             self.history_user = self.history_user[-5:]
    #             self.history_bot = self.history_bot[-5:]


if __name__ == "__main__":
    try:
        VoiceQANode()
    except rospy.ROSInterruptException:
        pass
