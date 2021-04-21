import sys

sys.path.append("/export/home1/NoCsBack/hci/mingxiao/Project/visual_dialog")
import os
import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import gmtime, strftime
from timeit import default_timer as timer
from torch.utils.data import DataLoader
from params import read_command_line

# from bert_base_dataloader import VDDataset
from bert_base_dataloader import VDDataset
from bert_base_net import VDModel
from config import Config
from bert_base_utils import (
    WarmupLinearScheduleNonZero,
    batch_iter,
    load_pretrained_weights_and_adjust_names,
)
from pytorch_transformers.optimization import AdamW
from utils.metrics import SparseGTMetrics, NDCG, scores_to_ranks
from torch.utils.tensorboard import SummaryWriter

train_info = open("one_round_model", "w")


def forward(visual_dialog, batch, params, sample_size=None, evaluation=False):
    tokens = batch["tokens"]
    mask = batch["mask"]
    padding_mask = batch["padding_mask"]
    token_type_ids = batch["segments"]

    orig_img_features = batch["image_feat"]
    orig_spatials = batch["image_loc"]
    orig_img_mask = batch["image_mask"]

    tokens = tokens.view(-1, tokens.shape[-1])
    mask = mask.view(-1, mask.shape[-1])
    padding_mask = padding_mask.view(-1, padding_mask.shape[-1])
    token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])

    features = orig_img_features.view(
        -1, orig_img_features.shape[-2], orig_img_features.shape[-1]
    )
    spatials = orig_spatials.view(-1, orig_spatials.shape[-2], orig_spatials.shape[-1])
    image_mask = orig_img_mask.view(-1, orig_img_mask.shape[-1])

    if sample_size:
        sample_indices = torch.randperm(tokens.shape[0])
        sample_indices = sample_indices[:sample_size]
    else:
        sample_indices = torch.arange(tokens.shape[0])

    tokens = tokens[sample_indices, :]
    mask = mask[sample_indices, :]
    padding_mask = padding_mask[sample_indices, :]
    token_type_ids = token_type_ids[sample_indices, :]

    features = features[sample_indices, :, :]
    spatials = spatials[sample_indices, :, :]
    image_mask = image_mask[sample_indices, :]

    next_sentence_labels = None
    image_target = None
    image_label = None

    if not evaluation:

        next_sentence_labels = batch["next_sentence_labels"]
        next_sentence_labels = next_sentence_labels.view(-1)
        next_sentence_labels = next_sentence_labels[sample_indices]
        next_sentence_labels = next_sentence_labels.to(params["device"])

        orig_img_target = batch["image_target"]
        orig_img_label = batch["image_label"]

        image_target = orig_img_target.view(
            -1, orig_img_target.shape[-2], orig_img_target.shape[-1]
        )
        image_label = orig_img_label.view(-1, orig_img_label.shape[-1])

        image_target = image_target[sample_indices, :, :]
        image_label = image_label[sample_indices, :]

        image_target = image_target.to(params["device"])
        image_label = image_label.to(params["device"])

    tokens = tokens.to(params["device"])
    mask = mask.to(params["device"])
    padding_mask = padding_mask.to(params["device"])
    features = features.to(params["device"])
    spatials = spatials.to(params["device"])
    image_mask = image_mask.to(params["device"])
    token_type_ids = token_type_ids.to(params["device"])
    image_mask = image_mask.float()
    total_padding_mask = torch.cat([padding_mask, image_mask], dim=1)

    nsp_loss = None
    loss = None
    lm_loss = None
    img_loss = None

    if not evaluation:
        img_loss, lm_loss, nsp_loss = visual_dialog(
            input_txt=tokens,
            input_img=features,
            img_loc=spatials,
            token_type_ids=token_type_ids,
            padding_mask=total_padding_mask,
            lm_label=mask,
            img_label=image_label,
            img_target=image_target,
            nsp_label=next_sentence_labels,
        )

        lm_loss = lm_loss.mean()
        nsp_loss = nsp_loss.mean()
        img_loss = img_loss.mean()

        loss = (
            (params["lm_loss_coeff"] * lm_loss)
            + (params["nsp_loss_coeff"] * nsp_loss)
            + (params["img_loss_coeff"] * img_loss)
        )

        return lm_loss, nsp_loss, img_loss, loss
    else:
        nsp_score = visual_dialog(
            input_txt=tokens,
            input_img=features,
            img_loc=spatials,
            token_type_ids=token_type_ids,
            padding_mask=total_padding_mask,
        )
        return nsp_score


def evaluate(visual_dialog, params, eval_batch_size, dataloader):
    sparse_metrics = SparseGTMetrics()
    ndcg = NDCG()

    visual_dialog.eval()
    batch_idx = 0
    with torch.no_grad():
        batch_size = 5
        for epoch_id, _, batch in batch_iter(dataloader, params):
            if epoch_id == 1:
                break
            tokens = batch["tokens"]
            padding_mask = batch["padding_mask"]
            mask = batch["mask"]
            token_type_ids = batch["token_type_ids"]
            gt_option_inds = batch["gt_option_inds"]
            gt_relevance = batch["gt_relevance"]
            gt_relevance_round_id = batch["round_id"].squeeze(1)

            num_rounds = tokens.shape[1]
            num_options = tokens.shape[2]

            tokens = tokens.view(-1, tokens.shape[-1])
            padding_mask = padding_mask.view(-1, padding_mask.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            mask = mask.view(-1, mask.shape[-1])

            features = batch["image_feat"]
            spatials = batch["image_loc"]
            image_mask = batch["image_mask"]
            max_num_regions = features.shape[-2]
            features = (
                features.unsqueeze(1)
                .unsqueeze(1)
                .expand(eval_batch_size, num_rounds, num_options, max_num_regions, 2048)
                .contiguous()
            )
            spatials = (
                spatials.unsqueeze(1)
                .unsqueeze(1)
                .expand(eval_batch_size, num_rounds, num_options, max_num_regions, 5)
                .contiguous()
            )
            image_mask = (
                image_mask.unsqueeze(1)
                .unsqueeze(1)
                .expand(eval_batch_size, num_rounds, num_options, max_num_regions)
                .contiguous()
            )

            features = features.view(-1, max_num_regions, 2048)
            spatials = spatials.view(-1, max_num_regions, 5)
            image_mask = image_mask.view(-1, max_num_regions)

            assert (
                tokens.shape[0]
                == padding_mask.shape[0]
                == token_type_ids.shape[0]
                == features.shape[0]
                == spatials.shape[0]
                == num_rounds * num_options * eval_batch_size
            )

            output = []
            print(eval_batch_size * num_rounds * num_options)
            for j in range(eval_batch_size * num_rounds * num_options // batch_size):
                # create chunks of the original batch
                item = {}
                item["tokens"] = tokens[j * batch_size : (j + 1) * batch_size, :]
                item["padding_mask"] = padding_mask[
                    j * batch_size : (j + 1) * batch_size, :
                ]
                item["token_type_ids"] = token_type_ids[
                    j * batch_size : (j + 1) * batch_size, :
                ]
                item["mask"] = mask[j * batch_size : (j + 1) * batch_size, :]

                item["image_feat"] = features[
                    j * batch_size : (j + 1) * batch_size, :, :
                ]
                item["image_loc"] = spatials[
                    j * batch_size : (j + 1) * batch_size, :, :
                ]
                item["image_mask"] = image_mask[
                    j * batch_size : (j + 1) * batch_size, :
                ]

                nsp_scores = forward(visual_dialog, item, params, evaluation=True)
                nsp_probs = F.softmax(nsp_scores, dim=1)
                assert nsp_probs.shape[-1] == 2
                output.append(nsp_probs[:, 0])

            output = torch.cat(output, 0).view(eval_batch_size, num_rounds, num_options)
            sparse_metrics.observe(output, gt_option_inds)
            output = output[torch.arange(output.size(0)), gt_relevance_round_id - 1, :]
            ndcg.observe(output, gt_relevance)
            batch_idx += 1

    visual_dialog.train()
    print("tot eval batches", batch_idx)
    all_metrices = {}
    all_metrices.update(sparse_metrics.retrieve(reset=True))
    all_metrices.update(ndcg.retrieve(reset=True))

    return all_metrices


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    writer = SummaryWriter("runs_one_round/loss")
    params = read_command_line()
    # params["continue"] = True
    # params["from_bilbo"] = False #"checkpoints/FromBilbo/visdial_dialog_15398.ckpt"
    # params["continue_path"] = "checkpoints/FromBilbo/visdial_dialog_161679.ckpt"
    # params["overfit"] = True
    # params["start_path"] = False
    # params["batch_size"] = 1
    #
    # make dir to save model
    os.makedirs("checkpoints", exist_ok=True)
    if not os.path.exists(params["save_path"]):
        os.mkdir(params["save_path"])
    pprint.pprint(params)
    print(params, file=train_info)

    # set dataset
    dataset = VDDataset(params)
    dataset.split = "train"

    batch_size = (
        params["batch_size"] // (params["sequences_per_image"])
        if (params["batch_size"] // (params["sequences_per_image"]))
        else 1
        if not params["overfit"]
        else 5
    )

    print("batch_size", batch_size)
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=params["num_workers"],
        drop_last=True,
        pin_memory=False,
    )

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params["device"] = device

    # build model
    config = Config()
    visual_dialog = VDModel(config)

    # optimization setting
    param_optimizer = list(visual_dialog.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in visual_dialog.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.1,
        },
        {
            "params": [
                p
                for n, p in visual_dialog.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=params["lr"])
    scheduler = WarmupLinearScheduleNonZero(
        optimizer, warmup_steps=30000, t_total=150000
    )

    start_iter_id = 0
    # load pretrained bert weights
    if params["start_path"]:
        if not params["continue"]:
            pretraiend_dict = load_pretrained_weights_and_adjust_names(
                params["start_path"]
            )
            model_dict = visual_dialog.state_dict()
            pretraiend_dict = {
                k: v for k, v in pretraiend_dict.items() if k in model_dict
            }
            print("number of keys transferred", len(pretraiend_dict))
            assert len(pretraiend_dict.keys()) > 0
            model_dict.update(pretraiend_dict)
            visual_dialog.load_state_dict(model_dict)
            del pretraiend_dict, model_dict
        else:
            print("continue to train")
            print(params["continue_path"])
            pretrained_dict = torch.load(params["continue_path"])
            model_dict = visual_dialog.state_dict()
            optimizer_dict = optimizer.state_dict()
            pretraiend_dict_model = pretrained_dict["model_state_dict"]
            pretraiend_dict_optimizer = pretrained_dict["optimizer_state_dict"]
            pretrained_dict_scheduler = pretrained_dict["scheduler_state_dict"]
            pretraiend_dict_model = {
                k: v for k, v in pretraiend_dict_model.items() if k in model_dict
            }
            pretraiend_dict_optimizer = {
                k: v
                for k, v in pretraiend_dict_optimizer.items()
                if k in optimizer_dict
            }
            model_dict.update(pretraiend_dict_model)
            optimizer_dict.update(pretraiend_dict_optimizer)
            visual_dialog.load_state_dict(model_dict)
            optimizer.load_state_dict(optimizer_dict)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            scheduler = WarmupLinearScheduleNonZero(
                optimizer,
                warmup_steps=700,
                t_total=20000,
                last_epoch=pretrained_dict["iter_id"],
            )
            scheduler.load_state_dict(pretrained_dict_scheduler)
            start_iter_id = pretrained_dict["iter_id"]

            del (
                pretrained_dict,
                pretraiend_dict_model,
                pretraiend_dict_optimizer,
                pretrained_dict_scheduler,
                model_dict,
                optimizer_dict,
            )
            torch.cuda.empty_cache()

    # set iteration number in epoch
    num_iter_epoch = dataset.numDataPoints["train"] // (
        params["batch_size"] // params["sequences_per_image"]
        if (params["batch_size"] // params["sequences_per_image"])
        else 1
        if not params["overfit"]
        else 5
    )
    # start_iter_id = 0
    print("\n%d iter per epoch ." % num_iter_epoch)

    # send model to gpu
    visual_dialog = nn.DataParallel(visual_dialog)
    visual_dialog.to(device)

    # training
    start_t = timer()
    optimizer.zero_grad()
    mlm_running_loss = 0
    nsp_running_loss = 0
    img_running_loss = 0
    running_loss = 0

    for epoch_id, idx, batch in batch_iter(train_dataloader, params):

        iter_id = start_iter_id + idx + (epoch_id * num_iter_epoch)
        visual_dialog.train()

        num_rounds = batch["tokens"].shape[1]
        num_samples = batch["tokens"].shape[2]

        # expand image_features
        orig_img_features = batch["image_feat"]
        orig_img_loc = batch["image_loc"]
        orig_img_label = batch["image_label"]
        orig_img_target = batch["image_target"]
        orig_img_mask = batch["image_mask"]

        features = (
            orig_img_features.unsqueeze(1)
            .unsqueeze(1)
            .expand(
                orig_img_features.shape[0],
                num_rounds,
                num_samples,
                orig_img_features.shape[1],
                orig_img_features.shape[2],
            )
            .contiguous()
        )
        spatials = (
            orig_img_loc.unsqueeze(1)
            .unsqueeze(1)
            .expand(
                orig_img_loc.shape[0],
                num_rounds,
                num_samples,
                orig_img_loc.shape[1],
                orig_img_loc.shape[2],
            )
            .contiguous()
        )
        image_label = (
            orig_img_label.unsqueeze(1)
            .unsqueeze(1)
            .expand(
                orig_img_label.shape[0],
                num_rounds,
                num_samples,
                orig_img_label.shape[1],
            )
            .contiguous()
        )
        image_target = (
            orig_img_target.unsqueeze(1)
            .unsqueeze(1)
            .expand(
                orig_img_target.shape[0],
                num_rounds,
                num_samples,
                orig_img_target.shape[1],
                orig_img_target.shape[2],
            )
            .contiguous()
        )
        image_mask = (
            orig_img_mask.unsqueeze(1)
            .unsqueeze(1)
            .expand(
                orig_img_mask.shape[0], num_rounds, num_samples, orig_img_mask.shape[1]
            )
            .contiguous()
        )

        batch["image_feat"] = features.contiguous()
        batch["image_loc"] = spatials.contiguous()
        batch["image_label"] = image_label.contiguous()
        batch["image_target"] = image_target.contiguous()
        batch["image_mask"] = image_mask.contiguous()

        # set sample size
        if params["overfit"]:
            sample_size = 2
        else:
            sample_size = params["batch_size"]
        loss = None
        lm_loss = None
        nsp_loss = None
        img_loss = None

        lm_loss, nsp_loss, img_loss, loss = forward(
            visual_dialog, batch, params, sample_size, evaluation=False
        )
        lm_nsp_loss = None
        if lm_loss is not None and nsp_loss is not None:
            lm_nsp_loss = lm_loss + nsp_loss

        loss /= params["batch_multiply"]
        loss.backward()

        mlm_running_loss += lm_loss.item()
        nsp_running_loss += nsp_loss.item()
        img_running_loss += img_loss.item()
        running_loss += loss.item()

        if iter_id % params["batch_multiply"] == 0 and iter_id > 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if iter_id % 100 == 0 and iter_id > 0:
            end_t = timer()
            cur_epoch = float(iter_id) / num_iter_epoch
            timestamp = strftime("%a %d %b %y %X", gmtime())

            print_lm_loss = mlm_running_loss / 100
            print_nsp_loss = nsp_running_loss / 100
            print_lm_nsp_loss = running_loss / 100
            print_img_loss = img_running_loss / 100

            print_format = "[%s][Ep: %.2f][Iter: %d][Time: %5.2fs][NSP + LM Loss: %.3g][LM Loss: %.3g][NSP Loss: %.3g][IMG Loss: %.3g]"
            print_info = [
                timestamp,
                cur_epoch,
                iter_id,
                end_t - start_t,
                print_lm_nsp_loss,
                print_lm_loss,
                print_nsp_loss,
                print_img_loss,
            ]

            print(print_format % tuple(print_info))
            print(print_format % tuple(print_info), file=train_info)
            start_t = end_t

            writer.add_scalar("lm_loss", print_lm_loss, iter_id)
            writer.add_scalar("img_loss", print_img_loss, iter_id)
            writer.add_scalar("nsp_loss", print_nsp_loss, iter_id)

            mlm_running_loss = 0
            nsp_running_loss = 0
            img_running_loss = 0
            running_loss = 0

        old_num_iter_epoch = num_iter_epoch
        if params["overfit"]:
            num_iter_epoch = 4
        if iter_id % num_iter_epoch == 0 and iter_id > 0:
            torch.save(
                {
                    "model_state_dict": visual_dialog.module.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "parameters": params,
                    "iter_id": iter_id,
                },
                os.path.join(params["save_path"], "visdial_dialog_%d.ckpt" % iter_id),
            )

        # fire evaluation
        if iter_id % (num_iter_epoch * 40) == 0 and iter_id > 0:
            print("fire evaluation", file=train_info)
            print("fire evaluation")
            eval_batch_size = 12
            if params["overfit"]:
                eval_batch_size = 5

            dataset.split = "val"
            val_dataloader = DataLoader(
                dataset,
                batch_size=eval_batch_size,
                shuffle=True,
                num_workers=params["num_workers"],
                drop_last=True,
                pin_memory=False,
            )
            all_metrices = evaluate(
                visual_dialog, params, eval_batch_size, val_dataloader
            )
            for metric_name, metric_value in all_metrices.items():
                print(f"{metric_name}: {metric_value}")
                print(f"{metric_name}: {metric_value}", file=train_info)

            dataset.split = "train"
        num_iter_epoch = old_num_iter_epoch
