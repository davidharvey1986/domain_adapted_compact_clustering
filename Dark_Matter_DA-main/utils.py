import argparse
import numpy as np
import torch
import wandb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import random


def parse_args():
    parser = argparse.ArgumentParser(
        description="Domain Adaptation for Astronomical Data"
    )

    # Data parameters
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--source_domain", type=str, default="bahamas")
    parser.add_argument("--target_domain", type=str, default="darkskies")
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--image_size", type=int, default=100)
    parser.add_argument("--use_log_transform", action="store_true", default=False)
    parser.add_argument("--use_normalization", action="store_true", default=False)

    # Model parameters
    parser.add_argument("--model", type=str, default="squeezenet1_1")
    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--cnn_base_channels", type=int, default=32)

    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="inverse_frequency",
        choices=["none", "inverse_frequency", "sqrt_inverse"],
    )

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.0003)
    parser.add_argument(
        "--optimizer", type=str, default="sgd", choices=["sgd", "adamw"]
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--use_nesterov", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--aug_h_flip_prob", type=float, default=0.5)
    parser.add_argument("--aug_v_flip_prob", type=float, default=0.5)
    parser.add_argument("--aug_rotation_prob", type=float, default=0.0)
    parser.add_argument("--aug_rotation_degrees", type=int, default=360)
    parser.add_argument("--aug_crop_prob", type=float, default=0.5)
    parser.add_argument("--aug_crop_scale_min", type=float, default=0.9)
    parser.add_argument("--aug_crop_scale_max", type=float, default=1.1)

    parser.add_argument(
        "--adaptation",
        type=str,
        default="none",
        choices=["none", "mmd", "coral", "dan", "dann", "cdan"],
    )
    parser.add_argument("--adaptation_weight", type=float, default=1.0)
    parser.add_argument(
        "--adaptation_schedule",
        type=str,
        default="dann",
        choices=["none", "dann", "linear"],
    )
    parser.add_argument("--adaptation_schedule_power", type=float, default=0.75)
    parser.add_argument("--adaptation_schedule_gamma", type=float, default=10.0)
    parser.add_argument("--mmd_kernel_mul", type=float, default=2.0)
    parser.add_argument("--mmd_kernel_num", type=int, default=5)
    parser.add_argument("--mmd_fix_sigma", type=float, default=None)
    parser.add_argument(
        "--domain_discriminator_hidden",
        type=int,
        default=256,
        help="Hidden size for domain discriminator",
    )

    parser.add_argument("--use_mixup", action="store_true", default=False)
    parser.add_argument(
        "--mixup_strategy", type=str, default="random", choices=["random", "same_index"]
    )
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument(
        "--mixup_per_pair",
        action="store_true",
        default=True,
        help="Use per-pair lambda values for more diversity",
    )

    # Logging parameters
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--project_name", type=str, default="Astro_clean_2")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--save_model", action="store_true", default=True)

    # Analysis parameters
    parser.add_argument("--analyze_predictions", action="store_true", default=True)

    args = parser.parse_args()

    if args.run_name is None:
        mixup_str = f"mixup_{args.mixup_strategy}" if args.use_mixup else "no_mixup"
        adapt_str = args.adaptation if args.adaptation != "none" else "no_adapt"
        pretrain_str = "pretrained" if args.pretrained else "scratch"
        args.run_name = f"{adapt_str}_{mixup_str}_{pretrain_str}_{args.source_domain}2{args.target_domain}_{args.model}"

    return args


def setup_wandb(args):
    if args.use_wandb:
        config = vars(args)
        wandb.init(project=args.project_name, name=args.run_name, config=config)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_classification_metrics(outputs, targets):
    preds = torch.argmax(outputs, dim=1)
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    accuracy = accuracy_score(targets_np, preds_np)
    f1 = f1_score(targets_np, preds_np, average="macro", zero_division=0)

    return {"accuracy": accuracy, "f1": f1}


def analyze_cross_section_predictions(
    all_outputs, all_cross_sections, all_binary_labels, prefix=""
):
    cross_sections_np = all_cross_sections.cpu().numpy()
    binary_labels_np = all_binary_labels.cpu().numpy()
    predictions = torch.argmax(all_outputs, dim=1).cpu().numpy()

    unique_cross_sections = np.unique(cross_sections_np)

    print(f"\n{prefix} - Predictions by Cross-Section Value:")
    print("=" * 60)

    overall_accuracy = np.mean(predictions == binary_labels_np)
    print(f"Overall Accuracy: {overall_accuracy:.3f}")
    print()

    for cross_section in sorted(unique_cross_sections):
        mask = cross_sections_np == cross_section
        cs_predictions = predictions[mask]
        cs_true_labels = binary_labels_np[mask]

        if len(cs_predictions) > 0:
            unique_preds, pred_counts = np.unique(cs_predictions, return_counts=True)
            pred_dict = dict(zip(unique_preds, pred_counts))

            unique_true, true_counts = np.unique(cs_true_labels, return_counts=True)
            true_dict = dict(zip(unique_true, true_counts))

            accuracy = np.mean(cs_predictions == cs_true_labels)

            print(
                f"Cross-section {cross_section:.3f} ({len(cs_predictions)} samples, acc: {accuracy:.3f}):"
            )
            print(
                f"  True labels: Class 0: {true_dict.get(0, 0)}, Class 1: {true_dict.get(1, 0)}"
            )
            print(
                f"  Predictions: Class 0: {pred_dict.get(0, 0)}, Class 1: {pred_dict.get(1, 0)}"
            )

            class_0_pct = pred_dict.get(0, 0) / len(cs_predictions) * 100
            class_1_pct = pred_dict.get(1, 0) / len(cs_predictions) * 100
            print(f"  Pred %: Class 0: {class_0_pct:.1f}%, Class 1: {class_1_pct:.1f}%")
            print()


def get_adaptation_schedule(iter_num, total_iters, args):
    if args.adaptation_schedule == "none":
        return args.adaptation_weight
    elif args.adaptation_schedule == "dann":
        p = float(iter_num) / total_iters
        alpha = 2.0 / (1.0 + np.exp(-args.adaptation_schedule_gamma * p)) - 1
        return alpha * args.adaptation_weight
    elif args.adaptation_schedule == "linear":
        p = float(iter_num) / total_iters
        return p * args.adaptation_weight
    else:
        return args.adaptation_weight


def inv_lr_scheduler(optimizer, iter_num, gamma, power, init_lr=0.001):
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return optimizer


def mixup_data(
    x,
    y,
    device,
    alpha=1.0,
    strategy="random",
    file_indices=None,
    dataset=None,
    per_pair=False,
):
    batch_size = x.size(0)

    if alpha > 0:
        if per_pair:
            lam = np.random.beta(alpha, alpha, size=batch_size)
            lam = torch.from_numpy(lam).float().to(device)
        else:
            lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    if strategy == "random":
        index = torch.randperm(batch_size).to(device)

        if per_pair and alpha > 0:
            lam_expanded = (
                lam.view(-1, 1, 1, 1) if len(x.shape) == 4 else lam.view(-1, 1)
            )
            mixed_x = lam_expanded * x + (1 - lam_expanded) * x[index, :]
        else:
            mixed_x = lam * x + (1 - lam) * x[index, :]

        y_a, y_b = y, y[index]

    elif strategy == "same_index" and file_indices is not None and dataset is not None:
        mixed_x = []
        y_a_list = []
        y_b_list = []
        lam_list = []
        successful_mixups = 0

        if hasattr(dataset, "dataset"):
            original_dataset = dataset.dataset
        else:
            original_dataset = dataset

        for i, (current_file_idx, current_img_idx) in enumerate(file_indices):
            candidates = original_dataset.get_same_position_candidates(
                current_file_idx, current_img_idx
            )

            if candidates:
                candidate_idx = random.choice(candidates)
                candidate_data = original_dataset[candidate_idx]

                candidate_img = candidate_data[0].to(device)
                if len(candidate_data) >= 3:
                    candidate_label = candidate_data[2].to(device)
                else:
                    candidate_label = candidate_data[1].to(device)

                if per_pair and alpha > 0:
                    lam_i = lam[i] if isinstance(lam, torch.Tensor) else lam
                else:
                    lam_i = lam

                mixed_img = lam_i * x[i] + (1 - lam_i) * candidate_img
                mixed_x.append(mixed_img)
                y_a_list.append(y[i])
                y_b_list.append(candidate_label)
                lam_list.append(lam_i)
                successful_mixups += 1
            else:
                mixed_x.append(x[i])
                y_a_list.append(y[i])
                y_b_list.append(y[i])
                lam_list.append(1.0)

        mixed_x = torch.stack(mixed_x)
        y_a = (
            torch.stack(y_a_list)
            if isinstance(y_a_list[0], torch.Tensor)
            else torch.tensor(y_a_list).to(device)
        )
        y_b = (
            torch.stack(y_b_list)
            if isinstance(y_b_list[0], torch.Tensor)
            else torch.tensor(y_b_list).to(device)
        )

        if per_pair:
            lam = torch.tensor(lam_list).to(device)

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    loss_a = criterion(pred, y_a)
    loss_b = criterion(pred, y_b)
    return (lam * loss_a + (1 - lam) * loss_b).mean()


def calculate_class_weights(data_loaders, args):
    class_counts = torch.zeros(2)

    for batch_data in data_loaders["source_train"]:
        if len(batch_data) >= 3:
            binary_labels = batch_data[2]
        else:
            continue
        unique, counts = torch.unique(binary_labels, return_counts=True)
        for class_idx, count in zip(unique, counts):
            class_counts[class_idx] += count

    print(f"Training class distribution:")
    print(f"  Class 0: {class_counts[0].int()} samples")
    print(f"  Class 1: {class_counts[1].int()} samples")

    total_samples = class_counts.sum()

    if args.weighting_scheme == "none":
        class_weights = torch.ones(2)
    elif args.weighting_scheme == "inverse_frequency":
        class_weights = total_samples / (2.0 * class_counts)
    elif args.weighting_scheme == "sqrt_inverse":
        inv_freq = total_samples / (2.0 * class_counts)
        class_weights = torch.sqrt(inv_freq)
    else:
        raise ValueError(f"Unknown weighting scheme: {args.weighting_scheme}")

    print(f"Weighting scheme: {args.weighting_scheme}")
    print(f"Class weights: [0: {class_weights[0]:.4f}, 1: {class_weights[1]:.4f}]")

    return class_weights
