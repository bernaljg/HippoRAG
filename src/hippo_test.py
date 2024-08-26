import argparse
from hipporag import HippoRAG

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--query', type=str)
    args = parser.parse_args()

    hipporag = HippoRAG(corpus_name=args.dataset)

    queries = [args.query]
    for query in queries:
        ranks, scores, logs = hipporag.rank_docs(query, top_k=10)

        print(ranks)
        print(scores)
        print(logs)