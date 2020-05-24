import logging
import numpy, random, time, json
import torch
import torch.nn.functional as F
import torch.optim as optim

from ssl_lib.algs.builder import gen_ssl_alg
from ssl_lib.algs import utils as alg_utils
from ssl_lib.models import utils as model_utils
from ssl_lib.consistency.builder import gen_consistency
from ssl_lib.models.builder import gen_model
from ssl_lib.datasets.builder import gen_dataloader
from ssl_lib.param_scheduler import scheduler
from ssl_lib.misc.meter import Meter


def evaluation(raw_model, eval_model, loader, device):
    raw_model.eval()
    eval_model.eval()
    sum_raw_acc = sum_acc = sum_loss = 0
    with torch.no_grad():
        for (data, labels) in loader:
            data, labels = data.to(device), labels.to(device)
            preds = eval_model(data)
            raw_preds = raw_model(data)
            loss = F.cross_entropy(preds, labels)
            sum_loss += loss.item()
            acc = (preds.max(1)[1] == labels).float().mean()
            raw_acc = (raw_preds.max(1)[1] == labels).float().mean()
            sum_acc += acc.item()
            sum_raw_acc += raw_acc.item()
    mean_raw_acc = sum_raw_acc / len(loader)
    mean_acc = sum_acc / len(loader)
    mean_loss = sum_loss / len(loader)
    raw_model.train()
    eval_model.train()
    return mean_raw_acc, mean_acc, mean_loss


def param_update(
    cfg,
    cur_iteration,
    model,
    teacher_model,
    optimizer,
    ssl_alg,
    consistency,
    labeled_data,
    ul_weak_data,
    ul_strong_data,
    labels,
    average_model
):
    start_time = time.time()

    all_data = torch.cat([labeled_data, ul_weak_data, ul_strong_data], 0)
    forward_func = model.forward
    stu_logits = forward_func(all_data)
    labeled_preds = stu_logits[:labeled_data.shape[0]]

    stu_unlabeled_weak_logits, stu_unlabeled_strong_logits = torch.chunk(stu_logits[labels.shape[0]:], 2, dim=0)

    if cfg.tsa:
        none_reduced_loss = F.cross_entropy(labeled_preds, labels, reduction="none")
        L_supervised = alg_utils.anneal_loss(
            labeled_preds, labels, none_reduced_loss, cur_iteration+1,
            cfg.iteration, labeled_preds.shape[1], cfg.tsa_schedule)
    else:
        L_supervised = F.cross_entropy(labeled_preds, labels)

    if cfg.coef > 0:
        # get target values
        if teacher_model is not None: # get target values from teacher model
            t_forward_func = teacher_model.forward
            tea_logits = t_forward_func(all_data)
            tea_unlabeled_weak_logits, _ = torch.chunk(tea_logits[labels.shape[0]:], 2, dim=0)
        else:
            t_forward_func = forward_func
            tea_unlabeled_weak_logits = stu_unlabeled_weak_logits

        # calc consistency loss
        model.update_batch_stats(False)
        y, targets, mask = ssl_alg(
            stu_preds = stu_unlabeled_strong_logits,
            tea_logits = tea_unlabeled_weak_logits.detach(),
            data = ul_strong_data,
            stu_forward = forward_func,
            tea_forward = t_forward_func
        )
        model.update_batch_stats(True)
        L_consistency = consistency(y, targets, mask, weak_prediction=tea_unlabeled_weak_logits.softmax(1))

    else:
        L_consistency = torch.zeros_like(L_supervised)        
        mask = None       

    # calc total loss
    coef = scheduler.exp_warmup(cfg.coef, cfg.warmup_iter, cur_iteration+1)
    loss = L_supervised + coef * L_consistency
    if cfg.entropy_minimization > 0:
        loss -= cfg.entropy_minimization * \
            (stu_unlabeled_weak_logits.softmax(1) * F.log_softmax(stu_unlabeled_weak_logits, 1)).sum(1).mean()

    # update parameters
    cur_lr = optimizer.param_groups[0]["lr"]
    optimizer.zero_grad()
    loss.backward()
    if cfg.weight_decay > 0:
        decay_coeff = cfg.weight_decay * cur_lr
        model_utils.apply_weight_decay(model.modules(), decay_coeff)
    optimizer.step()

    # update teacher parameters by exponential moving average
    if cfg.ema_teacher:
        model_utils.ema_update(
            teacher_model, model, cfg.ema_teacher_factor,
            cfg.weight_decay * cur_lr if cfg.ema_apply_wd else None, 
            cur_iteration if cfg.ema_teacher_warmup else None)
    # update evaluation model's parameters by exponential moving average
    if cfg.weight_average:
        model_utils.ema_update(
            average_model, model, cfg.wa_ema_factor, 
            cfg.weight_decay * cur_lr if cfg.wa_apply_wd else None)

    # calculate accuracy for labeled data
    acc = (labeled_preds.max(1)[1] == labels).float().mean()

    return {
        "acc": acc,
        "loss": loss.item(),
        "sup loss": L_supervised.item(),
        "ssl loss": L_consistency.item(),
        "mask": mask.float().mean().item() if mask is not None else 1,
        "coef": coef,
        "sec/iter": (time.time() - start_time)
    }


def main(cfg, logger):
    # set seed
    torch.manual_seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    random.seed(cfg.seed)
    # select device
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        logger.info("CUDA is NOT available")
        device = "cpu"
    # build data loader
    logger.info("load dataset")
    lt_loader, ult_loader, test_loader, num_classes, img_size = gen_dataloader(cfg.root, cfg.dataset, False, cfg, logger)

    # set consistency type
    consistency = gen_consistency(cfg.consistency, cfg)
    # set ssl algorithm
    ssl_alg = gen_ssl_alg(cfg.alg, cfg)
    # build student model
    model = gen_model(cfg.model, num_classes, img_size).to(device)
    # build teacher model
    if cfg.ema_teacher:
        teacher_model = gen_model(cfg.model, num_classes, img_size).to(device)
        teacher_model.load_state_dict(model.state_dict())
    else:
        teacher_model = None
    # for evaluation
    if cfg.weight_average:
        average_model = gen_model(cfg.model, num_classes, img_size).to(device)
        average_model.load_state_dict(model.state_dict())
    else:
        average_model = None

    model.train()

    logger.info(model)

    # build optimizer
    if cfg.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(), cfg.lr, cfg.momentum, weight_decay=0, nesterov=True
        )
    elif cfg.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(), cfg.lr, (cfg.momentum, 0.999), weight_decay=0
        )
    else:
        raise NotImplementedError
    # set lr scheduler
    if cfg.lr_decay == "cos":
        lr_scheduler = scheduler.CosineAnnealingLR(optimizer, cfg.iteration)
    elif cfg.lr_decay == "step":
        # TODO: fixed milstones
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [400000, ], cfg.lr_decay_rate)
    else:
        raise NotImplementedError

    # init meter
    metric_meter = Meter()
    test_acc_list = []
    raw_acc_list = []

    logger.info("training")
    for i, (l_data, ul_data) in enumerate(zip(lt_loader, ult_loader)):
        l_aug, labels = l_data
        ul_w_aug, ul_s_aug, _ = ul_data

        params = param_update(
            cfg, i, model, teacher_model, optimizer, ssl_alg,
            consistency, l_aug.to(device), ul_w_aug.to(device),
            ul_s_aug.to(device), labels.to(device),
            average_model
        )

        # moving average for reporting losses and accuracy
        metric_meter.add(params, ignores=["coef"])

        # display losses every cfg.disp iterations
        if ((i+1) % cfg.disp) == 0:
            state = metric_meter.state(
                header = f'[{i+1}/{cfg.iteration}]',
                footer = f'ssl coef {params["coef"]:.4g} | lr {optimizer.param_groups[0]["lr"]:.4g}'
            )
            logger.info(state)

        lr_scheduler.step()
        if ((i + 1) % cfg.checkpoint) == 0 or (i+1) == cfg.iteration:
            with torch.no_grad():
                if cfg.weight_average:
                    eval_model = average_model
                else:
                    eval_model = model
                logger.info("test")
                mean_raw_acc, mean_test_acc, mean_test_loss = evaluation(model, eval_model, test_loader, device)
                logger.info("test loss %f | test acc. %f | raw acc. %f", mean_test_loss, mean_test_acc, mean_raw_acc)
                test_acc_list.append(mean_test_acc)
                raw_acc_list.append(mean_raw_acc)

            torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model_checkpoint.pth"))
            torch.save(optimizer.state_dict(), os.path.join(cfg.out_dir, "optimizer_checkpoint.pth"))

    numpy.save(os.path.join(cfg.out_dir, "results"), test_acc_list)
    numpy.save(os.path.join(cfg.out_dir, "raw_results"), raw_acc_list)
    accuracies = {}
    for i in [1, 10, 20, 50]:
        logger.info("mean test acc. over last %d checkpoints: %f", i, numpy.median(test_acc_list[-i:]))
        logger.info("mean test acc. for raw model over last %d checkpoints: %f", i, numpy.median(raw_acc_list[-i:]))
        accuracies[f"last{i}"] = numpy.median(test_acc_list[-i:])

    with open(os.path.join(cfg.out_dir, "results.json"), "w") as f:
        json.dump(accuracies, f, sort_keys=True)


if __name__ == "__main__":
    import os, sys
    from parser import get_args
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # setup logger
    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    s_handler = logging.StreamHandler(stream=sys.stdout)
    s_handler.setFormatter(plain_formatter)
    s_handler.setLevel(logging.DEBUG)
    logger.addHandler(s_handler)
    f_handler = logging.FileHandler(os.path.join(args.out_dir, "console.log"))
    f_handler.setFormatter(plain_formatter)
    f_handler.setLevel(logging.DEBUG)
    logger.addHandler(f_handler)
    logger.propagate = False

    logger.info(args)

    main(args, logger)