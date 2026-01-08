#!/usr/bin/env python3

#!/usr/bin/env python3
import os, sys
import rospy
from bvi_fyp.srv import stt, sttRequest
from bvi_fyp.srv import tts, ttsRequest
from bvi_fyp.srv import rag, ragRequest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ui.ui_publisher import UIPublisher

class VoiceQANode:
    def __init__(self):
        rospy.init_node("voice_rag_node")

        self.ui = UIPublisher()
        self.ui.init()
        self.ui.publish_state(page="chat", status="ready")

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

        self.speak("I have successfully reached near the user.")
        self.ui.publish_chat("robot", "Dear user, I am your faculty guide robot. "
            "You can ask me for any faculty location related questions or ask for a navigation guide to your destination.")
        self.speak ("Dear user, I am your faculty guide robot. "
            "You can ask me for any faculty location related questions or ask for a navigation guide to your destination."
        )
        

        self.main_loop()

    def speak(self, text):
        try:
            self.tts_client(ttsRequest(text=text))
        except Exception as e:
            rospy.logerr(f"TTS call failed: {e}")

    def listen(self):
        try:
            print("Listening")
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

    # Nav ques detect
    def detect_navigation_intent(self, text):
        """
        Detect if the user is directly asking for navigation guidance
        (e.g. 'guide me to lab 1', 'take me to the library').

        Returns:
            (is_nav_intent: bool, destination: str)
        """
        if not text:
            return False, ""

        lower = text.lower()

        nav_phrases = [
            "guide me to",
            "take me to",
            "bring me to",
            "lead me to",
            "navigate to",
            "go to",
            "walk me to",
        ]

        for phrase in nav_phrases:
            if phrase in lower:
                idx = lower.find(phrase) + len(phrase)
                destination = text[idx:].strip(" .!?,")
                if not destination:
                    destination = text.strip()
                return True, destination

        return False, ""

    # Check whether RAG answers a navigation ques
    def answer_ends_with_guide_question(self, answer: str) -> bool:
        if not answer:
            return False
        trimmed = answer.strip()
        return trimmed.lower().endswith("me to guide you there?")

    # Extract dest from RAG answer
    def extract_destination_from_answer(self, raw_answer: str):
        """
        Look for a line like 'DESTINATION_TAG: <...>' in the answer.
        Returns (clean_answer_without_tag, destination_or_empty).
        """
        if not raw_answer:
            return raw_answer, ""

        lines = raw_answer.splitlines()
        dest = ""
        new_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith("destination_tag:"):
                parts = stripped.split(":", 1)
                if len(parts) == 2:
                    dest = parts[1].strip()
                # Don't include this line in the spoken answer
                continue
            else:
                new_lines.append(line)

        clean_answer = "\n".join(new_lines).strip()
        return clean_answer, dest
    
    def validate_location_with_rag(self, location):
        query = f"""
        You MUST answer using ONLY the retrieved context.
        Do NOT use any outside knowledge or assumptions.

        Task:
        Determine whether the retrieved context explicitly contains a location matching: "{location}"
        Matching can be exact or an obvious alias (e.g., "Tutorial 1" vs "Tutorial Room 1" vs "Bilik Tutorial 1" vs "BT1").

        Output format (exactly one line):
        VALID: YES
        or
        VALID: NO
        or
        VALID: UNKNOWN  (if the context does not mention it clearly)
        """

        resp = self.ask_rag(query)
        print(resp)
        return "yes" in resp.lower()

    def main_loop(self):
        # simple loop: listen → answer → speak
        # self.speak("Hello, I am your campus guide robot. You can ask me about locations on campus.")

        rate = rospy.Rate(0.1)  # e.g. one iteration every 10 seconds
        yes_words = ["yes", "yeah", "ya", "yup", "sure", "please", "ok", "okay"]
        no_words = ["no", "nope", "nah"]

        
        while not rospy.is_shutdown():
            self.ui.publish_chat("robot", "Please ask your question, or say 'exit' to stop.")
            self.speak("Please ask your question, or say 'exit' to stop.")
            
            got_yes = False

            user_text = self.listen()
            
            if not user_text:

                self.ui.publish_chat("robot", "Sorry, I did not hear anything.")
                self.speak("Sorry, I did not hear anything.")
                
                continue

            rospy.loginfo(f"User said: {user_text}")
            self.ui.publish_chat("user", user_text)

            # ---- exit / stop ----
            if user_text.lower() in ["exit", "quit", "stop"]:

                self.ui.publish_chat("robot", "Goodbye.")
                self.speak("Goodbye.")
                
                break

            # 1) Direct navigation command → nav mock + break
            is_nav, destination = self.detect_navigation_intent(user_text)
            if is_nav:
                rospy.loginfo(f"[NAV-INTENT] User requested navigation to: {destination}")

                # Location Validation
                if not self.validate_location_with_rag(destination):

                    self.ui.publish_chat("robot", f"Sorry, I couldn't find the location {destination}. "
                            "Please ask about campus locations or try again.")
                    self.speak(f"Sorry, I couldn't find the location {destination}. "
                            "Please ask about campus locations or try again.")
                    
                    continue

                # If valid → proceed
                print(f"[NAVIGATION MOCK] Starting navigation to: {destination}")

                self.ui.publish_chat("robot", f"Okay, I will guide you to {destination}.")
                self.speak(f"Okay, I will guide you to {destination}.")
                

                # call real nav service here later

                self.ui.publish_state(page="status", status="navigating", destination=destination)
                rospy.sleep(5)
                self.speak(f"We have successfully reached the {destination}.")
                self.ui.publish_state(page="status", status="navigation_completed", destination=destination)
                self.speak(f"Do you have more questions or need new navigation guide? Please answer with yes or no.")
                max_retries = 2
                confirm = None

                for attempt in range(max_retries):
                    confirm = self.listen()
                    confirm_lower = confirm.lower()
                    if any(w in confirm_lower for w in no_words):
                        self.speak("Okay, I will return to my starting position. Hope you are satisfied with my service. Goodbye!")

                        self.ui.publish_state(page="status", status="going_back")
                        rospy.sleep(5)
                        self.ui.publish_state(page="status", status="idle")
                        rate.sleep()
                        return
                    elif any(w in confirm_lower for w in yes_words):
                        self.speak("Sure! What do you need from me?")

                        got_yes = True
                        self.ui.publish_state(page="chat", status="ready")
                        break

                    else:
                        if attempt == 0:
                            self.speak("I did not hear a clear answer. Please answer with yes or no.")
                        else:
                            self.speak("I did not hear a clear yes. I will now return to my starting position. Hope you are satisfied with my service. Goodbye!")
                            rate.sleep()
                            return
              

            # 2) Otherwise, treat it as a question → call RAG
            raw_answer = self.ask_rag(user_text)
            rospy.loginfo(f"RAG raw answer: {raw_answer}")

            # Strip DESTINATION_TAG from the text and get destination from RAG
            answer, dest_from_rag = self.extract_destination_from_answer(raw_answer)
            rospy.loginfo(f"RAG cleaned answer: {answer}")
            rospy.loginfo(f"RAG destination tag: {dest_from_rag}")

            # Update history (limit length) with the CLEAN answer
            self.history_user.append(user_text)
            self.history_bot.append(answer)
            if len(self.history_user) > 5:
                self.history_user = self.history_user[-5:]
                self.history_bot = self.history_bot[-5:]

            # Speak RAG answer (without DESTINATION_TAG line)
            self.ui.publish_chat("robot", answer)
            self.speak(answer)
            
            # 3) If RAG ended with 'Do you need me to guide you there?' → yes/no branch
            if self.answer_ends_with_guide_question(answer):
                # clarify we want yes/no

                self.ui.publish_chat("robot", "Please answer with yes or no.")
                self.speak("Please answer with yes or no.")
                

                # rospy.sleep(1.0) 

                max_retries = 2
                confirm = None

                for attempt in range(max_retries):
                    confirm = self.listen()
                    if confirm:
                        break
                    rospy.loginfo(f"[NAV-CONFIRM] Empty result on attempt {attempt+1}")

                    self.ui.publish_chat("robot", "Sorry, I did not catch that. Please say yes or no.")
                    self.speak("Sorry, I did not catch that. Please say yes or no.")
                    

                    rospy.sleep(1.0)

                if not confirm:
                    rospy.loginfo("[NAV-CONFIRM] No response after retries; staying in Q&A mode.")

                    self.ui.publish_chat("robot", "I still did not hear a clear answer, so I will not start navigation. "
                        "Do you have more questions?")

                    self.speak(
                        "I still did not hear a clear answer, so I will not start navigation. "
                        "Do you have more questions?"
                    )
                    
                    rate.sleep()
                    continue

                confirm_lower = confirm.lower()
                rospy.loginfo(f"[NAV-CONFIRM] User said: {confirm}")
                self.ui.publish_chat("user", confirm)

                got_yes_after_nav = False

                
                if any(w in confirm_lower for w in yes_words):
                    # Prefer destination from RAG; fall back to user_text as last resort
                    if dest_from_rag and dest_from_rag.upper() != "NONE":
                        destination = dest_from_rag
                    else:
                        destination = user_text

                    rospy.loginfo(f"[NAV-START] Starting navigation (from RAG flow) to: {destination}")
                    print(f"[NAVIGATION MOCK] Starting navigation to: {destination}")
                    self.ui.publish_chat("robot", f"Okay, I will guide you to {destination} now.")
                    self.speak(f"Okay, I will guide you to {destination} now.")
                    
                    # call real nav service here
                    self.ui.publish_state(page="status", status="navigating", destination=destination)
                    rospy.sleep(5)
                    self.speak(f"We have successfully reached the {destination}.")
                    self.ui.publish_state(page="status", status="navigation_completed", destination=destination)

                    got_yes_after_nav = False
                    self.speak(f"Do you have more questions or need new navigation guide? Please answer with yes or no.")
                    max_retries = 2
                    confirm = None

                    for attempt in range(max_retries):
                        confirm = self.listen()
                        confirm_lower = confirm.lower()
                        if any(w in confirm_lower for w in no_words):
                            self.speak(f"Okay, I will return to my starting position. Hope you are satisfied with my service. Goodbye!")
                            self.ui.publish_state(page="status", status="going_back")
                            rospy.sleep(5)
                            self.ui.publish_state(page="status", status="idle")
                            rate.sleep()
                            return
                        elif any(w in confirm_lower for w in yes_words):
                            self.speak("Sure! What do you need from me?")
                            got_yes_after_nav = True
                            self.ui.publish_state(page="chat", status="ready")
                            rate.sleep()
                            break
                        else:
                            if attempt == 0:
                                self.speak("I did not hear a clear answer. Please answer with yes or no.")
                            else:
                                self.speak("I did not hear a clear yes. I will now return to my starting position. Hope you are satisfied with my service. Goodbye!")
                                self.ui.publish_state(page="status", status="idle")
                                rate.sleep()
                                return
                
                if got_yes_after_nav:
                    continue

                elif any(w in confirm_lower for w in no_words):
                    self.speak("Okay, I will not start navigation. Do you have more questions?")
                    self.ui.publish_chat("robot", "Okay, I will not start navigation. Do you have more questions?")
                    # loop back for next question
                    rate.sleep()
                    continue

                else:
                    # ambiguous answer → treat as no 
                    self.speak("I did not hear a clear yes, so I will not start navigation. Do you have more questions?")
                    self.ui.publish_chat("robot", "I did not hear a clear yes, so I will not start navigation. Do you have more questions?")
                    rate.sleep()
                    continue

            rate.sleep()


if __name__ == "__main__":
    try:
        VoiceQANode()
    except rospy.ROSInterruptException:
        pass
