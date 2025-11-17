#!/usr/bin/env python3

import rospy
from bvi_fyp.srv import rag, ragResponse
from rag_instructions import answer_query  # your module from step 1

class RagService:
    def __init__(self):
        rospy.init_node("rag_service_node")
        self.service = rospy.Service("/rag_query", rag, self.handle_rag)
        rospy.loginfo("RAG service /rag_query is ready.")

    def handle_rag(self, req):
        # Build chat_history list of tuples
        chat_history = list(zip(req.history_user, req.history_bot))

        try:
            answer = answer_query(req.query, chat_history)
            return ragResponse(answer=answer)
        except Exception as e:
            rospy.logerr(f"RAG error: {e}")
            return ragResponse(answer="I'm sorry, something went wrong while answering your question.")

if __name__ == "__main__":
    RagService()
    rospy.spin()
