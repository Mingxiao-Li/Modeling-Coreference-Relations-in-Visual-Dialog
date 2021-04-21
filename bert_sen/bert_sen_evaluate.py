import torch
import json
import os
import torch.nn as nn
import pprint
import torch.nn.functional as F
from torch.utils.data import DataLoader
from bert_sen_utils import batch_iter
from bert_sen_train import forward
from utils.metrics import scores_to_ranks, SparseGTMetrics, NDCG
from params import read_command_line
from bert_sen_net import VDModel
from bert_sen_dataloader import VDDataset
from config import Config

def evaluate(visual_dialog, params, eval_batch_size, dataloader):
    sparse_metrics = SparseGTMetrics()
    ndcg = NDCG()

    visual_dialog.eval()
    batch_idx = 0
    with torch.no_grad():
        batch_size = 50
        for epoch_id, _, batch in batch_iter(dataloader, params):
            if epoch_id == 1:
                break
            tokens = batch["tokens"]
            sentence_ids = batch["sentence_ids"]
            padding_mask = batch["padding_mask"]
            mask = batch["mask"]
            token_type_ids = batch["segments"]
            gt_option_inds = batch["gt_option_inds"]
            gt_relevance = batch["gt_relevance"]
            gt_relevance_round_id = batch["round_id"].squeeze(1)

            num_rounds = tokens.shape[1]
            num_options = tokens.shape[2]

            tokens = tokens.view(-1, tokens.shape[-1])
            sentence_ids = sentence_ids.view(-1, sentence_ids.shape[-1])
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
                item["sentence_ids"] = sentence_ids[
                    j * batch_size : (j + 1) * batch_size, :
                ]
                item["padding_mask"] = padding_mask[
                    j * batch_size : (j + 1) * batch_size, :
                ]
                item["segments"] = token_type_ids[
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


def eval_ai_generate(dataloader, visual_dialog, params, eval_batch_size, split="test"):
    ranks_json = []
    visual_dialog.eval()
    batch_idx = 0
    with torch.no_grad():
        batch_size = 100
        print("batch size for evaluation", batch_size)
        for epochId, idx, batch in batch_iter(dataloader, params):
            if epochId == 1:
                break
            print("num_finished : ", idx * eval_batch_size)
            tokens = batch["tokens"]
            num_rounds = tokens.shape[1]
            num_options = tokens.shape[2]
            tokens = tokens.view(-1, tokens.shape[-1])
            mask = batch["mask"]
            mask = mask.view(-1, mask.shape[-1])
            token_type_ids = batch["segments"]
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            sentence_ids = batch["sentence_ids"]
            sentence_ids = sentence_ids.view(-1, sentence_ids.shape[-1])
            padding_mask = batch["padding_mask"]
            padding_mask = padding_mask.view(-1, padding_mask.shape[-1])
            # attn_mask = batch["attn_mask"]
            # attn_mask = attn_mask.view(-1,attn_mask.shape[-1])

            # get image features
            features = batch["image_feat"]
            spatials = batch["image_loc"]
            image_mask = batch["image_mask"]

            # expand the image features to match those of tokens
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
                == mask.shape[0]
                == token_type_ids.shape[0]
                == sentence_ids.shape[0]
                == features.shape[0]
                == spatials.shape[0]
                == image_mask.shape[0]
                == num_rounds * num_options * eval_batch_size
            )

            output = []
            assert (eval_batch_size * num_rounds * num_options) // batch_size == (
                eval_batch_size * num_rounds * num_options
            ) / batch_size
            for j in range((eval_batch_size * num_rounds * num_options) // batch_size):
                item = {}
                item["tokens"] = tokens[j * batch_size : (j + 1) * batch_size, :]
                item["segments"] = token_type_ids[
                    j * batch_size : (j + 1) * batch_size, :
                ]
                item["mask"] = mask[j * batch_size : (j + 1) * batch_size, :]
                item["padding_mask"] = padding_mask[
                    j * batch_size : (j + 1) * batch_size
                ]
                # item["attn_mask"] = attn_mask[j*batch_size:(j+1)*batch_size]
                item["sentence_ids"] = sentence_ids[
                    j * batch_size : (j + 1) * batch_size
                ]

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
                # normalize nsp scpres
                nsp_probs = F.softmax(nsp_scores, dim=1)
                assert nsp_probs.shape[-1] == 2
                output.append(nsp_probs[:, 0])

            output = torch.cat(output, 0).view(eval_batch_size, num_rounds, num_options)
            ranks = scores_to_ranks(output)
            ranks = ranks.squeeze(1)
            for i in range(eval_batch_size):
                ranks_json.append(
                    {
                        "image_id": batch["image_id"][i].item(),
                        "round_id": int(batch["round_id"][i].item()),
                        "ranks": [rank.item() for rank in ranks[i][:]],
                    }
                )
            batch_idx += 1
    return ranks_json


def evaluate_test_set(dataloader, model, params, eval_batch_size, split):
    ranks_json = eval_ai_generate(
        dataloader, visual_dialog, params, eval_batch_size, split
    )
    json.dump(ranks_json, open(params["save_name"] + "language_predictions.txt", "w"))


def evaluate_validation_set(dataloader, model, params, eval_batch_size):
    all_metrices = evaluate(model, params, eval_batch_size, dataloader)
    for metric_name, metric_value in all_metrices.items():
        print(f"{metric_name}: {metric_value}")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    params = read_command_line()
    params[
        "start_path"
    ] = "checkpoints/09-Sep-20-13:58:48-Wed_9733833/visdial_dialog_338756.ckpt"  # visdial_dialog_215572.ckpt"
    params["visdial_processed_val"] = "../../data/visdial/core_val/visdial_val_co4.json"
    pprint.pprint(params)
    dataset = VDDataset(params)
    eval_batch_size = 5
    split = "val"
    # split = "test"
    dataset.split = split
    dataloader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=params["num_workers"],
        drop_last=True,
        pin_memory=False,
    )
    print("ldata_loader_len", len(dataloader))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params["device"] = device
    config = Config()
    visual_dialog = VDModel(config)

    if params["start_path"]:
        pretrained_dict = torch.load(params["start_path"])

        if "model_state_dict" in pretrained_dict:
            pretrained_dict = pretrained_dict["model_state_dict"]

        model_dict = visual_dialog.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print("number of keys transferred", len(pretrained_dict))
        model_dict.update(pretrained_dict)
        visual_dialog.load_state_dict(model_dict)

    visual_dialog = nn.DataParallel(visual_dialog)
    visual_dialog.to(device)
    # evaluate_test_set(dataloader,visual_dialog,params,eval_batch_size,split=split)
    evaluate_validation_set(dataloader, visual_dialog, params, eval_batch_size)
