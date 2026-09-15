import argparse
from src.config import TrainingConfig
from src.training.trainer import train
from src.utils.checkpoint import find_latest_checkpoint


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Korean LLM Advanced v3 Training Script")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--max-steps", type=int, default=50000, help="Maximum training steps")
    parser.add_argument("--accumulation-steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=200, help="Warmup steps")
    parser.add_argument("--eval-interval", type=int, default=10000, help="Evaluation and checkpoint interval")
    parser.add_argument("--resume", type=str, default="auto", help="'auto', 'latest', or path to checkpoint")
    parser.add_argument("--download-datasets", action="store_true", help="Force redownload of datasets")
    parser.add_argument("--samples-per-dataset", type=int, default=None, help="Limit samples per dataset for debugging")

    args = parser.parse_args()

    resume_ckpt = None
    if args.resume == "auto":
        resume_ckpt = 'latest' if find_latest_checkpoint() else None
    elif args.resume.lower() == "latest":
        resume_ckpt = 'latest'
    elif args.resume:
        resume_ckpt = args.resume

    return TrainingConfig(
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        eval_interval=args.eval_interval,
        resume_from_checkpoint=resume_ckpt,
        download_datasets=args.download_datasets,
        samples_per_dataset=args.samples_per_dataset,
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)
