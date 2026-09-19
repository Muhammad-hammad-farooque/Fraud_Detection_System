"""
Train the shipped model on realistic synthetic data (T-18).

The model used to be fitted on 20 hand-written rows that were perfectly
separable on amount alone (A9). It is now trained the way every future model
will be: through scripts/retrain.py, on point-in-time features computed by the
same queries serving uses, with a chronological train/test split, and promoted
only if it beats the active model on the same held-out window.

The training data comes from scripts/generate_data.py and lives in its own
database, never the application's, so synthetic labels cannot mix with real
ones.

Usage:
    python -m app.ML.train_model                       # generate 50k rows if needed, then train
    python -m app.ML.train_model --regenerate          # replace the synthetic data first
    python -m app.ML.train_model --transactions 5000   # a quicker, smaller run
"""
import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.generate_data import DEFAULT_DATABASE, GeneratorConfig, generate   # noqa: E402  (sets env first)

from sqlalchemy import create_engine                                            # noqa: E402
from sqlalchemy.orm import sessionmaker                                         # noqa: E402

from app import models                                                          # noqa: E402
from app.database import Base                                                   # noqa: E402
from scripts.retrain import run                                                 # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the fraud model on synthetic data.")
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--transactions", type=int, default=GeneratorConfig.transactions)
    parser.add_argument("--users", type=int, default=GeneratorConfig.users)
    parser.add_argument("--seed", type=int, default=GeneratorConfig.seed)
    parser.add_argument("--regenerate", action="store_true", help="replace existing synthetic data")
    parser.add_argument("--no-activate", action="store_true", help="register a winning model without activating it")
    args = parser.parse_args(argv)

    if args.database.startswith("sqlite:///"):
        os.makedirs(os.path.dirname(args.database[len("sqlite:///"):]) or ".", exist_ok=True)
    engine = create_engine(args.database)
    Base.metadata.create_all(bind=engine)

    with sessionmaker(bind=engine)() as db:
        if args.regenerate or db.query(models.Transaction).count() == 0:
            started = time.time()
            report = generate(db, GeneratorConfig(transactions=args.transactions, users=args.users,
                                                  seed=args.seed), reset=True)
            print(f"Generated {report.transactions} transactions ({report.fraud_rate:.2%} fraud) "
                  f"in {time.time() - started:.0f}s: {dict(report.patterns)}")

        started = time.time()
        result = run(db=db, activate_on_win=not args.no_activate)

    print(f"\nTraining finished in {time.time() - started:.0f}s: {result.reason}")
    if result.challenger_metrics:
        m = result.challenger_metrics
        print(f"Challenger  AUC {m['auc']:.4f}  PR-AUC {m['average_precision']:.4f}  "
              f"recall@1%FPR {m['recall_at_1pct_fpr']:.2%}")
    if result.champion_metrics.get("average_precision") is not None:
        m = result.champion_metrics
        print(f"Champion    AUC {m['auc']:.4f}  PR-AUC {m['average_precision']:.4f}  "
              f"recall@1%FPR {m['recall_at_1pct_fpr']:.2%}")
    if result.manifest:
        print(f"Registered {result.manifest.version}")
    return 0 if result.challenger_metrics else 1


if __name__ == "__main__":
    sys.exit(main())
