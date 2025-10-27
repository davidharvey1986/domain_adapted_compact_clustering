import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os

from dataset import prepare_dataloaders
from model import create_model
from train import evaluate, train_epoch
from utils import parse_args, set_seed, setup_wandb, calculate_class_weights


def main() -> None:
    args = parse_args()
    setup_wandb(args)
    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(
        f"Model: {args.model} ({'pretrained' if args.pretrained else 'from scratch'})"
    )
    print(f"Mixup: {args.mixup_strategy if args.use_mixup else 'Disabled'}")
    print(f"Adaptation: {args.adaptation}")
    print(f"Experiment: {args.run_name}")

    data_loaders = prepare_dataloaders(args)
    model = create_model(args).to(device)

    if args.use_wandb:
        wandb.watch(model, log_freq=100)

    class_weights = calculate_class_weights(data_loaders, args)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    if args.use_mixup:
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer_params = [
        {"params": model.backbone.parameters(), "lr": args.lr},
        {"params": model.classification_head.parameters(), "lr": args.lr},
    ]

    if (
        args.adaptation == "cdan"
        and model.multilinear_map is not None
        and hasattr(model.multilinear_map, "parameters")
    ):
        optimizer_params.append(
            {"params": model.multilinear_map.parameters(), "lr": args.lr}
        )

    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            optimizer_params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.use_nesterov,
        )
    else:
        optimizer = optim.AdamW(
            optimizer_params, lr=args.lr, weight_decay=args.weight_decay
        )

    discriminator_optimizer = None

    if args.adaptation in ["dann", "cdan"] and model.domain_discriminator is not None:
        discriminator_params = [
            {"params": model.domain_discriminator.parameters(), "lr": args.lr}
        ]

        if args.optimizer == "sgd":
            discriminator_optimizer = optim.SGD(
                discriminator_params,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov=args.use_nesterov,
            )
        else:
            discriminator_optimizer = optim.AdamW(
                discriminator_params, lr=args.lr, weight_decay=args.weight_decay
            )

        print("Created separate optimizer for domain discriminator")

    best_target_accuracy = 0.0
    best_target_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = train_epoch(
            model,
            data_loaders["source_train"],
            data_loaders["target_train"],
            optimizer,
            criterion,
            device,
            epoch,
            args,
            discriminator_optimizer,
        )

        print(
            f"Train - Total: {train_metrics['train_total_loss']:.4f}, "
            + f"Task: {train_metrics['train_task_loss']:.4f}, "
            + f"Adapt: {train_metrics['train_adaptation_loss']:.4f}"
        )

        if args.adaptation in ["dann", "cdan"]:
            print(
                f"Train - Discriminator: {train_metrics['train_discriminator_loss']:.4f}, "
                + f"LR: {train_metrics['learning_rate']:.6f}"
            )
        else:
            print(f"Train - LR: {train_metrics['learning_rate']:.6f}")

        if args.use_wandb:
            wandb.log({"epoch": epoch, **train_metrics}, step=epoch)

        if epoch % args.eval_interval == 0:
            print(f"Evaluating at epoch {epoch}...")

            source_val_results = evaluate(
                model,
                data_loaders["source_val"],
                criterion,
                device,
                args,
                prefix="source_val",
            )
            target_test_results = evaluate(
                model,
                data_loaders["target_test"],
                criterion,
                device,
                args,
                prefix="target_test",
            )

            current_target_accuracy = target_test_results.get("target_test_accuracy", 0)
            current_target_f1 = target_test_results.get("target_test_f1", 0)

            if current_target_accuracy > best_target_accuracy:
                best_target_accuracy = current_target_accuracy
            if current_target_f1 > best_target_f1:
                best_target_f1 = current_target_f1

            log_metrics = {
                "source_val_accuracy": source_val_results.get("source_val_accuracy", 0),
                "source_val_f1": source_val_results.get("source_val_f1", 0),
                "target_test_accuracy": target_test_results.get(
                    "target_test_accuracy", 0
                ),
                "target_test_f1": target_test_results.get("target_test_f1", 0),
                "best_target_accuracy": best_target_accuracy,
                "best_target_f1": best_target_f1,
            }

            if args.use_wandb:
                wandb.log(log_metrics, step=epoch)

    print(f"\n{'=' * 80}")
    print(f"TRAINING COMPLETED")
    print(f"{'=' * 80}")

    if args.save_model:
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{args.run_name}_final.pth"

        save_dict = {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": args,
        }

        if discriminator_optimizer is not None:
            save_dict["discriminator_optimizer_state_dict"] = (
                discriminator_optimizer.state_dict()
            )

        torch.save(save_dict, model_path)
        print(f"Saved final model to {model_path}")

    print(f"\nFINAL EVALUATION:")
    source_val_results = evaluate(
        model,
        data_loaders["source_val"],
        criterion,
        device,
        args,
        prefix="source_final",
    )
    target_test_results = evaluate(
        model,
        data_loaders["target_test"],
        criterion,
        device,
        args,
        prefix="target_final",
    )

    print(f"\nSUMMARY:")
    print(f"  Experiment: {args.run_name}")
    print(
        f"  Model: {args.model} ({'pretrained' if args.pretrained else 'from scratch'})"
    )
    print(f"  Domain: {args.source_domain} → {args.target_domain}")
    print(f"  Adaptation: {args.adaptation}")
    print(f"  Mixup: {args.mixup_strategy if args.use_mixup else 'None'}")

    final_target_accuracy = target_test_results.get("target_final_accuracy", 0)
    final_target_f1 = target_test_results.get("target_final_f1", 0)
    print(f"  Final Target Accuracy: {final_target_accuracy:.4f}")
    print(f"  Final Target F1: {final_target_f1:.4f}")
    print(f"  Best Target Accuracy: {best_target_accuracy:.4f}")
    print(f"  Best Target F1: {best_target_f1:.4f}")

    if args.use_wandb:
        final_metrics = {
            "final_source_accuracy": source_val_results.get("source_final_accuracy", 0),
            "final_source_f1": source_val_results.get("source_final_f1", 0),
            "final_target_accuracy": target_test_results.get(
                "target_final_accuracy", 0
            ),
            "final_target_f1": target_test_results.get("target_final_f1", 0),
            "best_target_accuracy_overall": best_target_accuracy,
            "best_target_f1_overall": best_target_f1,
        }
        wandb.log(final_metrics)
        wandb.finish()


if __name__ == "__main__":
    main()
