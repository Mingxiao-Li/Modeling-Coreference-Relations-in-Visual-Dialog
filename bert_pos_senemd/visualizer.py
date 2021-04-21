from bert_pos_senemd_net import VDModel
from bert_pos_senemd_dataloader import VDDataset
from params import read_command_line
from config import Config
from torch.utils import data
from vis_utils import draw_attn
from pytorch_transformers.tokenization_bert import BertTokenizer
import matplotlib.pyplot as plt
import torch
import os

# 0,1,2,27
def forward(model, batch, params):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens = batch["tokens"]
    mask = batch["mask"]
    padding_mask = batch["padding_mask"]
    token_type_ids = batch["segments"]
    pos_tag_ids = batch["pos_tag_ids"]
    pos_tag_target = batch["pos_tag_target"]
    sentence_ids = batch["sentence_ids"]

    features = batch["image_feat"]
    spatials = batch["image_loc"]
    img_mask = batch["image_mask"]

    tokens = tokens.view(-1, tokens.shape[-1])
    pos_tag_ids = pos_tag_ids.view(-1, pos_tag_ids.shape[-1])
    pos_tag_target = pos_tag_target.view(-1, pos_tag_target.shape[-1])
    mask = mask.view(-1, mask.shape[-1])
    padding_mask = padding_mask.view(-1, padding_mask.shape[-1])
    token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
    sentence_ids = sentence_ids.view(-1, sentence_ids.shape[-1])

    tokens = tokens.to(params["device"])
    pos_tag_ids = pos_tag_ids.to(params["device"])
    pos_tag_target = pos_tag_target.to(params["device"])
    mask = mask.to(params["device"])
    padding_mask = padding_mask.to(params["device"])
    features = features.to(params["device"])
    spatials = spatials.to(params["device"])
    image_mask = img_mask.to(params["device"])
    token_type_ids = token_type_ids.to(params["device"])
    sentence_ids = sentence_ids.to(params["device"])
    total_padding_mask = torch.cat([padding_mask, image_mask], dim=1)

    _, attention_weights = model(
        input_txt=tokens,
        input_pos=pos_tag_ids,
        input_img=features,
        sentence_pos=sentence_ids,
        img_loc=spatials,
        token_type_ids=token_type_ids,
        padding_mask=total_padding_mask,
    )
    print("attn_shape", len(attention_weights))
    for k, attentions in enumerate(attention_weights):
        # print(list(tokens[0].numpy()))
        token_x = tokenizer.convert_ids_to_tokens(list(tokens[0].numpy()))
        # print(token_x)
        attn_head = attentions[0]
        print("plotting")
        f, axes = plt.subplots(4, 4, figsize=(50, 50))
        attn_head = attn_head.squeeze(0)
        for i in range(4):
            for j in range(4):
                attn_map = attn_head[i * 4 + j, 255:265, 88:114]
                axes[i][j] = draw_attn(
                    token_x[88:114],
                    ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                    attn_map.detach().numpy(),
                    f,
                    axes[i][j],
                )
        plt.tight_layout()
        plt.savefig("attention_map_img" + str(k))
        plt.clf()


def get_specific_round(data_loader, data_id, round_id, answer_option):
    for i, batch in enumerate(data_loader):
        model.eval()
        item = {}
        item["image_feat"] = batch["image_feat"]
        item["image_loc"] = batch["image_loc"]
        item["image_label"] = batch["image_label"]
        item["image_mask"] = batch["image_mask"]
        item["sentence_ids"] = batch["sentence_ids"][:, round_id, answer_option, :]
        item["pos_tag_target"] = batch["pos_tag_target"][:, round_id, answer_option, :]
        item["tokens"] = batch["tokens"][:, round_id, answer_option, :]
        item["mask"] = batch["mask"][:, round_id, answer_option, :]
        #        item["next_sentence_labels"] = batch["next_sentence_labels"][:,-1,1]
        item["padding_mask"] = batch["padding_mask"][:, round_id, answer_option, :]
        item["segments"] = batch["segments"][:, round_id, answer_option, :]
        item["pos_tag_ids"] = batch["pos_tag_ids"][:, round_id, answer_option, :]
        if i == data_id:
            break
    print(batch["image_id"])
    print(batch["image_loc"])
    print(answer_option)
    return item


def draw_attention_on_graph(img_id="VisualDialog_val2018_000000284024.jpg"):
    path = "../VisualDialog_val2018/" + img_id


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    config = Config()
    model = VDModel(config)
    param = read_command_line()
    param["path"] = "../model_checkpoints/visdial_pos_sen.ckpt"
    param["visdial_processed_val"] = "../../data/visdial/core_val/visdial_val_co6.json"

    dataset = VDDataset(param)

    dataset.split = "val"
    batch_size = 1
    data_loader = data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=param["num_workers"],
        drop_last=True,
        pin_memory=False,
    )

    device = torch.device("cpu")
    param["device"] = device
    pretrained_dict = torch.load(param["path"], map_location="cpu")
    model_dict = model.state_dict()
    pretrained_dict_model = pretrained_dict["model_state_dict"]
    pretrained_dict_model = {
        k: v for k, v in pretrained_dict_model.items() if k in model_dict
    }
    model_dict.update(pretrained_dict_model)
    model.to(device)

    del pretrained_dict_model, pretrained_dict
    item = get_specific_round(data_loader, 9, 7, 5)
    forward(model, item, param)
