from ragen.args import build_argument_parser
from ragen.client import OpenAIClient, PgClient, generate_prompt
from ragen.emb import Embedding
from ragen.file import Chunk, ChunkGenerator


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    gen = ChunkGenerator(args.data, args.chunk_size)
    embedding = Embedding(args.emb_model)
    chat_client = OpenAIClient(args.api_key, args.api_base)
    OpenAIClient(args.emb_api_key, args.emb_api_base)
    PgClient(args.db_host, args.db_user, args.db_password, args.db_port)
    context = []
    for i, chunk in enumerate(gen.generate()):
        context.append(Chunk(index=i, text=chunk, emb=embedding.encode(chunk)))

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
        chat_client.chat(
            generate_prompt(user_context, request), args.model, args.api_key
        )
