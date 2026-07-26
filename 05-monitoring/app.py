import streamlit as st

from assistant import create_assistant
from db_save import save_conversation
from db_feedback import save_feedback
from judge import evaluate_relevance

assistant = create_assistant()

st.title("Course Assistant")

user_input = st.text_input("Ask a question about the course:")

if st.button("Get Answer"):
    with st.spinner("Processing ..."):
        if user_input:
            answer = assistant.rag(user_input)
            st.success("Completed!")
            st.write(answer)

            record = assistant.last_call
            st.write("Metrics:")
            st.write(f"Response Time: {record.response_time:.2f} seconds")
            st.write(f"Prompt Tokens: {record.prompt_tokens}")
            st.write(f"Completion Tokens: {record.completion_tokens}")
            st.write(f"Cost: ${record.cost:.4f}")

            conversation_id = save_conversation(record, user_input, "llm-zoomcamp")
            st.session_state.conversation_id = conversation_id

            ## LLM as a judge for answer relevance
            relevance, explanation = evaluate_relevance(user_input, answer)
            save_feedback(conversation_id, "judge",
                relevance=relevance, explanation=explanation)
            st.write(f"Relevance: {relevance}")
            st.write(f"Explanation: {explanation}")

            ## Feedback buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("+1"):
                    cid = st.session_state.conversation_id
                    save_feedback(cid, "user", score=1)
                    st.write("Thanks!")

            with col2:
                if st.button("-1"):
                    cid = st.session_state.conversation_id
                    save_feedback(cid, "user", score=-1)
                    st.write("Thanks for the feedback!")
        else:
            st.write("Please enter a question.")