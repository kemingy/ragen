from ragen.args import build_argument_parser
from ragen.emb import Embedding
from ragen.file import Chunk, ChunkGenerator
from ragen.llm import ask_llm, generate_prompt


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    # print(args)

    # print("The following files will be processed:", args.data)
    gen = ChunkGenerator(args.data, args.chunk_size)
    embedding = Embedding(args.emb_model)
    context = []
    for chunk in gen.generate():
        context.append(Chunk(text=chunk, emb=embedding.encode(chunk)))

    print("Welcome to Ragen!")
    print(
        "You are using the {} embedding model and {} language model.".format(
            args.emb_model,
            args.model,
        )
    )
    print("You're in the context of the following files:", args.data)
    print("End the conversation by pressing `Ctrl+C`")
    while True:
        try:
            user_input = input("Enter your question: ")
        except KeyboardInterrupt:
            print("\nBye!")
            break
        request = Chunk(text=user_input, emb=embedding.encode(user_input))
        user_context = embedding.retrieve(request, context, args.top_k)
        ask_llm(generate_prompt(user_context, request), args.model, args.api_key)
