import warnings

from losses import RFCMLoss, MSLoss
import torch
import matplotlib.pyplot as plt


def train(cfg, pred_model, optimizer, train_loader, chkpt, poly_decay=None):
    device = cfg.SYSTEM.DEVICE
    known_losses = ['RFCM', 'MS']
    assert cfg.SOLVER.LOSS in known_losses, "Only able to handle {} loss functions, but {} was given.".format(known_losses, cfg.SOLVER.LOSS)
    if cfg.SOLVER.LOSS == 'RFCM':
        print("Using RFCM loss")
        loss_fn = RFCMLoss(fuzzy_factor=cfg.SOLVER.FUZZY_FACTOR, regularizer_wt=cfg.SOLVER.REGULARIZER, chkpt=chkpt)
    elif cfg.SOLVER.LOSS == 'MS':
        loss_fn = MSLoss(reg=cfg.SOLVER.REGULARIZER, chkpt=chkpt)
        print("Using MS loss")

    for epoch in range(cfg.SOLVER.EPOCHS):
        pred_model.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_loader):
            display_loss = batch_idx % cfg.SYSTEM.LOG_FREQ == 0
            loss_fn.set_display(display_loss)
            y_pred = pred_model(data['input'].to(device))  # data['input'] is the input data, (Batch, 1, H, W)
            pred_loss = loss_fn(y_pred, data['output'].to(device))  # data['output'] is the input used for clustering loss (Batch, 1, H, W)
            pred_loss.backward()
            if ((batch_idx + 1) % cfg.SOLVER.ITER_SIZE == 0) or ((batch_idx + 1) == len(train_loader)):  # code for batch accumulation from https://stackoverflow.com/questions/68479235/cuda-out-of-memory-error-cannot-reduce-batch-size (accessed on 24th of Sept, 2021)
                optimizer.step()
                optimizer.zero_grad()

            if display_loss:
                loss_str = "epoch: {}, batch_idx: {}, loss: {}".format(epoch, batch_idx, pred_loss)
                chkpt.log_training(loss_str)
                print(loss_str)

        if poly_decay:
            poly_decay.step()
            print("lr: {}".format({optimizer.param_groups[0]['lr']}))


def get_maximum_activation(pred):
    classes = torch.argmax(pred, dim=1).to(torch.long)
    num_classes = pred.shape[1]
    binary_classes = torch.zeros_like(pred)
    for c in range(num_classes):
        binary_classes[:, c, :] = (classes==c).to(pred.dtype)

    return binary_classes


def precision(pred, gt):
    intersection = pred * gt
    return torch.count_nonzero(intersection, dim=(-2, -1))/torch.count_nonzero(pred, dim=(-2, -1))


def recall(pred, gt):
    intersection = pred * gt
    return torch.count_nonzero(intersection, dim=(-2, -1))/torch.count_nonzero(gt, dim=(-2, -1))


def binary_jaccard(pred, gt):
    i = torch.count_nonzero(pred * gt, dim=(-2, -1))
    u = torch.count_nonzero(pred + gt, dim=(-2, -1))
    return i / u


def dice_index(pred, gt):
    numerator = torch.count_nonzero(pred*gt, dim=(-2, -1))
    denominator = torch.count_nonzero(pred, dim=(-2, -1)) + torch.count_nonzero(gt, dim=(-2, -1))
    return (2*numerator)/denominator


def update_metrics_dict(metrics, metric_name, fn, pred, gt):
    if metric_name not in metrics.keys():
        metrics[metric_name] = fn(pred, gt)
    else:
        metrics[metric_name] = torch.cat((metrics[metric_name], fn(pred, gt)), dim=0)

    return metrics


def test(cfg, model, loader, test_type, chkpt, samples=None):
    device = cfg.SYSTEM.DEVICE
    with torch.inference_mode():
        metrics = {}
        metrics_names = ['jaccard', 'dice', 'precision', 'recall']
        metrics_fn = [binary_jaccard, dice_index, precision, recall]
        for batch_idx, data in enumerate(loader):
            if test_type!="quantitative" and samples and data['im_name'][0][:-4] in samples:
                save_imgs = True
            elif test_type!="quantitative" and not samples:
                warnings.warn('No samples provided. Saving qualitative results for the whole testing set')
                save_imgs = True
            else:
                save_imgs = False

            print("Batch idx {}".format(batch_idx))
            model.eval()
            y = data['output'].to(device) # binary ground truth
            pred = model(data['input'].to(device))
            activations = get_maximum_activation(pred)
            if test_type!="quantitative" and save_imgs:
                image_filename = data['im_name'][0][:-4] + '.png'
                chkpt.save_img_tensor(data['input'][0, 0], image_filename, grayscale=True, remove_xy_ticks=True)
                for c in range(activations.shape[1]):
                    chkpt.save_img_tensor(activations[0, c], 'class_'+ str(c) + '_' + image_filename, grayscale=True,
                                          remove_xy_ticks=True)
                    chkpt.save_img_tensor(pred[0, c], 'prediction_' + str(c) + '_' + image_filename, grayscale=True,
                                          remove_xy_ticks=True)

                chkpt.save_img_tensor(data['output'][0, 0], 'gt_' + image_filename, grayscale=True, remove_xy_ticks=True)

            if test_type!="qualitative":
                for name, fn in zip(metrics_names, metrics_fn):
                    metrics = update_metrics_dict(metrics, name, fn, activations, y)

        if test_type!="qualitative":
            for metric_name, results in metrics.items():
                metric_str = "{} mean  {} +- {} std\n".format(metric_name, torch.mean(results, dim=0), torch.std(results, unbiased=True, dim=0))
                chkpt.log_testing(metric_str)
                print(metric_str)

