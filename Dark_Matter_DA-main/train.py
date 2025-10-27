import torch
import numpy as np
from adaptation import get_adaptation_loss
from utils import calculate_classification_metrics, inv_lr_scheduler, mixup_data, mixup_criterion, analyze_cross_section_predictions, get_adaptation_schedule
import torch.nn.functional as F

def train_epoch(model, source_loader, target_loader, optimizer, criterion, device, epoch, args, discriminator_optimizer=None):
    model.train()

    train_task_loss = 0.0
    train_adaptation_loss = 0.0
    train_total_loss = 0.0
    train_discriminator_loss = 0.0

    len_source = len(source_loader)
    len_target = len(target_loader)
    num_batches = max(len_source, len_target)

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    source_dataset = source_loader.dataset.dataset if hasattr(source_loader.dataset, 'dataset') else source_loader.dataset

    for batch_idx in range(num_batches):
        try:
            source_batch = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            source_batch = next(source_iter)

        source_data, source_cross_sections, source_binary_labels, source_file_indices, source_img_indices = source_batch
        source_file_info = list(zip(source_file_indices.numpy(), source_img_indices.numpy()))
        source_data = source_data.to(device)
        source_cross_sections = source_cross_sections.to(device)
        source_binary_labels = source_binary_labels.to(device)

        optimizer = inv_lr_scheduler(
            optimizer, batch_idx + epoch * num_batches, args.gamma,
            power=args.adaptation_schedule_power, init_lr=args.lr
        )
        
        total_iters = args.epochs * num_batches
        current_iter = batch_idx + epoch * num_batches
        adaptation_weight = get_adaptation_schedule(current_iter, total_iters, args)
        
        if args.adaptation in ["dann", "cdan"]:
            model.update_grl_alpha(adaptation_weight)

        if args.use_mixup:
            mixed_source_data, source_binary_labels_a, source_binary_labels_b, lam = mixup_data(
                source_data, source_binary_labels, device=device, alpha=args.mixup_alpha,
                strategy=args.mixup_strategy, file_indices=source_file_info, dataset=source_dataset,
                per_pair=args.mixup_per_pair
            )

            source_outputs = model(mixed_source_data)
            source_features = source_outputs["features"]
        
            source_classification = source_outputs["classification"]
            task_loss = mixup_criterion(
                criterion, source_classification, 
                source_binary_labels_a, source_binary_labels_b, lam
            )

        else:
            source_outputs = model(source_data)
            source_features = source_outputs["features"]
            
            source_classification = source_outputs["classification"]
            task_loss = criterion(source_classification, source_binary_labels)



        adaptation_loss = torch.tensor(0.0, device=device)
        discriminator_loss = torch.tensor(0.0, device=device)
        
        if args.adaptation != 'none':
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)

            target_data, _, _, _, _ = target_batch
            target_data = target_data.to(device)
            target_outputs = model(target_data)
            target_features = target_outputs["features"]
            
            if args.adaptation in ["dann", "cdan"]:
                if discriminator_optimizer is not None:
                    discriminator_optimizer = inv_lr_scheduler(
                        discriminator_optimizer, batch_idx + epoch * num_batches, args.gamma,
                        power=args.adaptation_schedule_power, init_lr=args.lr
                    )
                    
                    discriminator_optimizer.zero_grad()

                    if args.adaptation == "dann":
                        domain_pred_source = model.domain_discriminator(source_features.detach())
                        domain_pred_target = model.domain_discriminator(target_features.detach())
                        
                    elif args.adaptation == "cdan":
                        pred_source = F.softmax(source_outputs["classification"].detach(), dim=1)
                        pred_target = F.softmax(target_outputs["classification"].detach(), dim=1)
                        
                        mapped_source = model.multilinear_map(source_features.detach(), pred_source)
                        mapped_target = model.multilinear_map(target_features.detach(), pred_target)
                        
                        domain_pred_source = model.domain_discriminator(mapped_source)
                        domain_pred_target = model.domain_discriminator(mapped_target)
                    
                    domain_label_source = torch.ones_like(domain_pred_source)
                    domain_label_target = torch.zeros_like(domain_pred_target)

                    discriminator_loss = F.binary_cross_entropy_with_logits(domain_pred_source, domain_label_source) + \
                                    F.binary_cross_entropy_with_logits(domain_pred_target, domain_label_target)
                    
                    discriminator_loss.backward()
                    discriminator_optimizer.step()
                
                optimizer.zero_grad()
                
                if args.adaptation == "dann":
                    reversed_features_source = model.grl(source_features)
                    reversed_features_target = model.grl(target_features)
                    
                    domain_pred_source_grl = model.domain_discriminator(reversed_features_source)
                    domain_pred_target_grl = model.domain_discriminator(reversed_features_target)
                    
                elif args.adaptation == "cdan":
                    pred_source = F.softmax(source_outputs["classification"], dim=1)
                    pred_target = F.softmax(target_outputs["classification"], dim=1)
                    
                    mapped_source = model.multilinear_map(source_features, pred_source)
                    mapped_target = model.multilinear_map(target_features, pred_target)
                    
                    reversed_mapped_source = model.grl(mapped_source)
                    reversed_mapped_target = model.grl(mapped_target)
                    
                    domain_pred_source_grl = model.domain_discriminator(reversed_mapped_source)
                    domain_pred_target_grl = model.domain_discriminator(reversed_mapped_target)
                
                domain_label_source_grl = torch.ones_like(domain_pred_source_grl)  
                domain_label_target_grl = torch.zeros_like(domain_pred_target_grl)  
                
                adaptation_loss = F.binary_cross_entropy_with_logits(domain_pred_source_grl, domain_label_source_grl) + \
                                F.binary_cross_entropy_with_logits(domain_pred_target_grl, domain_label_target_grl)

            else:
                optimizer.zero_grad()
                adaptation_kwargs = {}
                
                adaptation_loss = get_adaptation_loss(
                    source_features, target_features, args, **adaptation_kwargs
                )
                adaptation_loss = adaptation_weight * adaptation_loss
        else:
            optimizer.zero_grad()

        total_loss = task_loss + adaptation_loss

        total_loss.backward()
        optimizer.step()

        train_task_loss += task_loss.item()
        train_adaptation_loss += (adaptation_loss.item() if isinstance(adaptation_loss, torch.Tensor) else adaptation_loss)
        train_total_loss += total_loss.item()
        train_discriminator_loss += (discriminator_loss.item() if isinstance(discriminator_loss, torch.Tensor) else discriminator_loss)

    return {
        "train_task_loss": train_task_loss / num_batches,
        "train_adaptation_loss": train_adaptation_loss / num_batches,
        "train_total_loss": train_total_loss / num_batches,
        "train_discriminator_loss": train_discriminator_loss / num_batches,
        "learning_rate": optimizer.param_groups[0]["lr"]
    }

def evaluate(model, data_loader, criterion, device, args, prefix="val"):
    model.eval()
    eval_loss = 0
    all_outputs = []
    all_targets = []
    all_cross_sections = []
    all_binary_labels = []

    with torch.no_grad():
        for batch_data in data_loader:
            data, cross_sections, binary_labels = batch_data[:3]
            binary_labels = binary_labels.to(device)
            targets = binary_labels
            all_cross_sections.append(cross_sections.cpu())
            all_binary_labels.append(binary_labels.cpu())

            data = data.to(device)
            outputs_dict = model(data)

            outputs = outputs_dict["classification"]
            loss = criterion(outputs, targets).mean()
                
            eval_loss += loss.item()
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_cross_sections = torch.cat(all_cross_sections, dim=0)

    avg_loss = eval_loss / len(data_loader)
    
    all_binary_labels = torch.cat(all_binary_labels, dim=0)
    metrics = calculate_classification_metrics(all_outputs, all_targets)
    print(f"{prefix} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    
    if args.analyze_predictions:
        analyze_cross_section_predictions(all_outputs, all_cross_sections, all_binary_labels, prefix)
    
    results = {f"{prefix}_loss": avg_loss}
    for k, v in metrics.items():
        results[f"{prefix}_{k}"] = v

    return results