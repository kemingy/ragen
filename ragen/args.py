from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


def build_argument_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Ragen: retrieval augmented generation for text",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        help="data source, support glob pattern like `**/*.txt`",
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--model",
        help="language model name",
        default="gpt-3.5-turbo",
    )
    parser.add_argument(
        "--emb-model",
        help=(
            "embedding model name, find more at "
            "https://huggingface.co/spaces/mteb/leaderboard"
        ),
        default="thenlper/gte-base",
    )
    parser.add_argument(
        "--emb-dim",
        help="embedding dimension",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--api-key",
        help="language model API key",
    )
    parser.add_argument(
        "--api-base",
        help="language model API base URL",
    )
    parser.add_argument(
        "--emb-api-key",
        help="embedding model API key",
    )
    parser.add_argument(
        "--emb-api-base",
        help="embedding model API base URL",
    )
    parser.add_argument(
        "--top-k",
        help="top k candidates to be retrieved",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--chunk-size",
        help="length of each chunk",
        default=300,
        type=int,
    )
    parser.add_argument(
        "--db-host",
        help="database host",
        default="localhost",
    )
    parser.add_argument(
        "--db-port",
        help="database port",
        default=5432,
        type=int,
    )
    parser.add_argument(
        "--db-user",
        help="database user",
        default="postgres",
    )
    parser.add_argument(
        "--db-password",
        help="database password",
        default="",
    )

    return parser
