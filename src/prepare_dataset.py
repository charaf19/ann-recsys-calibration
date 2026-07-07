import argparse
from datasets.movielens import prepare_movielens
from datasets.goodbooks import prepare_goodbooks
from datasets.amazon_books import prepare_amazon_books

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ml-1m","ml-20m","goodbooks","amazon-books"], required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    if args.dataset == "ml-1m": p = prepare_movielens("1m", args.out)
    elif args.dataset == "ml-20m": p = prepare_movielens("20m", args.out)
    elif args.dataset == "goodbooks": p = prepare_goodbooks(args.out)
    elif args.dataset == "amazon-books": p = prepare_amazon_books(args.out)
    print(f"[prepare_dataset] wrote {p}")

if __name__ == "__main__":
    main()
