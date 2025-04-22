# def generate_answer(query: str, retrieved_docs: list, generator, max_length: int = 450) -> str:
#     """
#     Generate an answer based on the user query and the retrieved documents.
    
#     The function builds a prompt that includes the recall documents and the user question,
#     then uses the generator model to produce a comprehensive answer.
#     """
#     context = "\n\n".join(retrieved_docs)
#     prompt = (
#         "You are an expert in vehicle safety and recall information. "
#         "Below are details from various vehicle recall records including summaries, models, and links for more information:\n\n"
#         f"{context}\n\n"
#         "Using the above recall data and your domain knowledge, answer the following question:\n"
#         f"Question: {query}\n\n"
#         "Answer:"
#     )
#     results = generator(prompt, max_length=max_length, num_return_sequences=1)
#     answer = results[0]['generated_text']
#     return answer

from models import generate_chat_completion

def generate_answer(query: str, retrieved_docs: list, max_tokens: int = 300) -> str:
    context = "\n\n".join(retrieved_docs[:3])  # use top 3 docs
    prompt = (
        "The following are summaries of vehicle recall reports:\n\n"
        f"{context}\n\n"
        f"Based on the above information, answer the following question:\n"
        f"{query}\n\n"
        "Answer:"
    )
    return generate_chat_completion(prompt, max_tokens=max_tokens)
