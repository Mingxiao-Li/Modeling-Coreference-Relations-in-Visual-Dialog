import torch
import json
import random
import numpy as np
from torch.utils import data
from utils.image_feature_reader import ImageFeaturesH5Reader
from pytorch_transformers.tokenization_bert import BertTokenizer

class VDDataset(data.Dataset):
    def __init__(self, params):
        self.numDataPoints = {}
        num_samples_train = params["num_train_samples"]
        num_samples_val = params["num_val_samples"]
        self._image_features_reader = ImageFeaturesH5Reader(
            params["visdial_image_feats"]
        )
        with open(params["visdial_processed_train"]) as f:
            self.visdial_data_train = json.load(f)
            if params["overfit"]:
                if num_samples_train:
                    self.numDataPoints["train"] = num_samples_train
                else:
                    self.numDataPoints["train"] = 5
            else:
                if num_samples_train:
                    self.numDataPoints["train"] = num_samples_train
                else:
                    self.numDataPoints["train"] = len(
                        self.visdial_data_train["data"]["dialogs"]
                    )

        with open(params["visdial_processed_val"]) as f:
            self.visdial_data_val = json.load(f)
            if params["overfit"]:
                if num_samples_val:
                    self.numDataPoints["val"] = num_samples_val
                else:
                    self.numDataPoints["val"] = 5
            else:
                if num_samples_val:
                    self.numDataPoints["val"] = num_samples_val
                else:
                    self.numDataPoints["val"] = len(
                        self.visdial_data_val["data"]["dialogs"]
                    )

        with open(params["visdial_processed_test"]) as f:
            self.visdial_data_test = json.load(f)
            self.numDataPoints["test"] = len(self.visdial_data_test["data"]["dialogs"])

        self.overfit = params["overfit"]
        with open(params["visdial_processed_val_dense_annotations"]) as f:
            self.visdial_data_val_dense = json.load(f)

        self.num_options = params["num_options"]
        self._split = "train"
        self.subsets = ["train", "val", "test"]
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer = tokenizer
        tokens = ["[CLS]", "[SEP]", "[MASK]"]
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
        self.CLS = indexed_tokens[0]
        self.SEP = indexed_tokens[1]
        self.MASK = indexed_tokens[2]
        self.params = params
        self._max_region_num = 37

    def __len__(self):
        return self.numDataPoints[self._split]

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, split):
        assert split in self.subsets
        self._split = split

    def encode_input(self, utterances, MASK, CLS, SEP, max_seq_len, mask_prob=0.15):
        def list2tensorpad(input_list, max_seq_len):
            input_tensor = torch.LongTensor([input_list])
            if input_tensor.shape[1] >= max_seq_len:
                return input_tensor[:, :max_seq_len]
            input_tensor_zeros = torch.zeros(1, max_seq_len, dtype=torch.long)
            input_tensor_zeros[0, : input_tensor.shape[1]] = input_tensor
            input_tensor = input_tensor_zeros
            return input_tensor

        # process text
        token_id_list = []
        masked_token_list = []
        segment_id_list = []
        cur_segment = 0

        token_id_list.append(CLS)
        masked_token_list.append(0)
        segment_id_list.append(0)

        for i, cur_utterance in enumerate(utterances):
            cur_masked_index = [
                1 if random.random() < mask_prob else 0
                for _ in range(len(cur_utterance))
            ]
            masked_token_list.extend(cur_masked_index)
            token_id_list.extend(cur_utterance)
            segment_id_list.extend(([cur_segment] * len(cur_utterance)))

            token_id_list.append(SEP)
            segment_id_list.append(cur_segment)
            masked_token_list.append(0)
            cur_segment = cur_segment ^ 1

        dialog_length = len(token_id_list)
        # sentence padding 37->image
        padding_mask = torch.ones(1, max_seq_len)
        padding_mask[0, :dialog_length] = 0

        assert len(masked_token_list) == len(token_id_list)
        tokens = list2tensorpad(token_id_list, max_seq_len)
        masked_tokens = list2tensorpad(masked_token_list, max_seq_len)
        segments = list2tensorpad(segment_id_list, max_seq_len)

        masked_tokens[0, masked_tokens[0, :] == 0] = -1
        mask = masked_tokens[0, :] == 1
        masked_tokens[0, mask] = tokens[0, mask]
        tokens[0, mask] = MASK

        return tokens, segments, masked_tokens, padding_mask

    def encode_img(
        self, features, num_boxes, boxes, image_target, max_regions=37, mask_prob=0.15
    ):
        # process image
        output_label = []
        num_boxes = min(int(num_boxes), max_regions)
        mix_boxes_pad = np.zeros((max_regions, boxes.shape[-1]))
        mix_features_pad = np.zeros((max_regions, features.shape[-1]))
        mix_image_target = np.zeros((max_regions, image_target.shape[-1]))

        mix_boxes_pad[:num_boxes] = boxes[:num_boxes]
        mix_features_pad[:num_boxes] = features[:num_boxes]
        mix_image_target[:num_boxes] = image_target[:num_boxes]

        boxes = mix_boxes_pad
        features = mix_features_pad
        image_target = mix_image_target

        for i in range(num_boxes):
            prob = random.random()
            if prob < mask_prob:
                prob /= mask_prob
                # %80 randomly change token to mask token
                if prob < 0.9:
                    features[i] = 0
                output_label.append(1)
            else:
                output_label.append(-1)

        image_mask = [0] * (int(num_boxes))
        while len(image_mask) < max_regions:
            image_mask.append(1)
            output_label.append(-1)

        output_label[random.randint(1, len(output_label) - 1)] = 1
        image_label = torch.LongTensor(output_label)
        image_label[0] = 0
        image_mask = torch.tensor(image_mask).float()

        features = torch.tensor(features).float()
        img_loc = torch.tensor(boxes).float()
        image_target = torch.tensor(image_target).float()

        return features, img_loc, image_label, image_target, image_mask

    def __getitem__(self, index):
        def tokens2str(seq):
            dialog_sequence = ""
            for sentence in seq:
                for word in sentence:
                    dialog_sequence += self.tokenizer._convert_id_to_token(word) + " "
                dialog_sequence += "</end>"
            dialog_sequence = dialog_sequence.encode("utf8")
            return dialog_sequence

        def pruneRounds(context, num_rounds):
            start_segment = 1
            len_context = len(context)
            cur_rounds = (len(context) // 2) + 1
            l_index = 0
            if cur_rounds > num_rounds:
                # caption is not part of the final input
                l_index = len_context - (2 * num_rounds)
                start_segment = 0
            return context[l_index:], start_segment

        MAX_SEQ_LEN = self.params["max_seq_len"] - 1
        cur_data = None
        if self._split == "train":
            cur_data = self.visdial_data_train["data"]
        elif self._split == "val":
            cur_data = self.visdial_data_val["data"]
        else:
            cur_data = self.visdial_data_test["data"]

        num_options = self.num_options
        assert num_options > 1 and num_options <= 100

        dialog = cur_data["dialogs"][index]
        cur_questions = cur_data["questions"]
        cur_answers = cur_data["answers"]
        img_id = dialog["image_id"]

        if self._split == "train":
            utterances = []
            utterances_random = []
            tokenized_caption = self.tokenizer.encode(dialog["caption"])

            utterances.append([tokenized_caption])  # positive sample
            utterances_random.append([tokenized_caption])  # negative sample
            tot_len = len(tokenized_caption) + 1

            for rnd, utterance in enumerate(dialog["dialog"]):
                cur_rnd_utterance = utterances[-1].copy()
                cur_rnd_utterance_random = utterances[-1].copy()

                tokenized_question = self.tokenizer.encode(
                    cur_questions[utterance["question"]] + " ?"
                )
                tokenized_answer = self.tokenizer.encode(
                    cur_answers[utterance["answer"]]
                )
                cur_rnd_utterance.append(tokenized_question)
                cur_rnd_utterance.append(tokenized_answer)

                question_len = len(tokenized_question)
                answer_len = len(tokenized_answer)
                tot_len += question_len
                tot_len += answer_len + 1  # for sep

                cur_rnd_utterance_random.append(
                    self.tokenizer.encode(cur_questions[utterance["question"]] + " ?")
                )
                utterances.append(cur_rnd_utterance)
                num_inds = len(utterance["answer_options"])
                gt_option_ind = utterance["gt_index"]

                negative_samples = []
                for _ in range(self.params["num_negative_samples"]):
                    all_inds = list(range(num_inds))
                    all_inds.remove(gt_option_ind)
                    all_inds = all_inds[: (num_options - 1)]
                    tokenized_random_utterance = None
                    option_ind = None

                    while len(all_inds):
                        option_ind = random.choice(all_inds)
                        tokenized_random_utterance = self.tokenizer.encode(
                            cur_answers[utterance["answer_options"][option_ind]]
                        )
                        if MAX_SEQ_LEN >= (
                            tot_len + len(tokenized_random_utterance) + 1
                        ):
                            break
                        else:
                            all_inds.remove(option_ind)
                    if len(all_inds) == 0:
                        tokenized_random_utterance = tokenized_random_utterance[
                            :answer_len
                        ]
                    t = cur_rnd_utterance_random.copy()
                    t.append(tokenized_random_utterance)
                    negative_samples.append(t)
                utterances_random.append(negative_samples)
            utterances = utterances[1:]
            utterances_random = utterances_random[1:]
            assert len(utterances) == len(utterances_random) == 10

            tokens_all_rnd = []
            masked_all_rnd = []
            next_labels_all_rnd = []
            padding_mask_all_rnd = []
            segments_all_rnd = []

            for j, context in enumerate(utterances):
                tokens_all = []
                masked_all = []
                next_labels_all = []
                padding_mask_all = []
                segments_all = []

                context, _ = pruneRounds(context, self.params["visdial_tot_rounds"])
                tokens, segment, mask, padding_mask = self.encode_input(
                    context,
                    self.MASK,
                    self.CLS,
                    self.SEP,
                    max_seq_len=MAX_SEQ_LEN,
                    mask_prob=self.params["mask_prob"],
                )
                tokens_all.append(tokens)
                masked_all.append(mask)
                padding_mask_all.append(padding_mask)
                segments_all.append(segment)
                next_labels_all.append(torch.LongTensor([0]))

                for context_random in utterances_random[j]:
                    context_random, _ = pruneRounds(
                        context_random, self.params["visdial_tot_rounds"]
                    )
                    (
                        tokens_random,
                        segment,
                        mask_random,
                        padding_mask,
                    ) = self.encode_input(
                        context_random,
                        self.MASK,
                        self.CLS,
                        self.SEP,
                        max_seq_len=MAX_SEQ_LEN,
                        mask_prob=self.params["mask_prob"],
                    )
                    tokens_all.append(tokens_random)
                    masked_all.append(mask_random)
                    padding_mask_all.append(padding_mask)
                    segments_all.append(segment)
                    next_labels_all.append(torch.LongTensor([1]))

                tokens_all_rnd.append(torch.cat(tokens_all, 0).unsqueeze(0))
                masked_all_rnd.append(torch.cat(masked_all, 0).unsqueeze(0))
                next_labels_all_rnd.append(torch.cat(next_labels_all, 0).unsqueeze(0))
                padding_mask_all_rnd.append(torch.cat(padding_mask_all, 0).unsqueeze(0))
                segments_all_rnd.append(torch.cat(segments_all, 0).unsqueeze(0))

            tokens_all_rnd = torch.cat(tokens_all_rnd, 0)
            masked_all_rnd = torch.cat(masked_all_rnd, 0)
            next_labels_all_rnd = torch.cat(next_labels_all_rnd, 0)
            padding_mask_all_rnd = torch.cat(padding_mask_all_rnd, 0)
            segments_all_rnd = torch.cat(segments_all_rnd, 0)

            item = {}
            item["tokens"] = tokens_all_rnd
            item["mask"] = masked_all_rnd
            item["next_sentence_labels"] = next_labels_all_rnd
            item["padding_mask"] = padding_mask_all_rnd
            item["segments"] = segments_all_rnd

            # read image feature
            features, num_boxes, boxes, _, image_target = self._image_features_reader[
                img_id
            ]
            features, img_loc, image_label, image_target, image_mask = self.encode_img(
                features, num_boxes, boxes, image_target
            )
            item["image_feat"] = features
            item["image_loc"] = img_loc
            item["image_label"] = image_label
            item["image_target"] = image_target
            item["image_mask"] = image_mask
            return item

        elif self.split == "val":

            ge_relevance = None
            utterances = []
            gt_option_inds = []
            utterances.append([self.tokenizer.encode(dialog["caption"])])
            options_all = []

            for rnd, utterance in enumerate(dialog["dialog"]):
                cur_rnd_utterance = utterances[-1].copy()
                cur_rnd_utterance.append(
                    self.tokenizer.encode(cur_questions[utterance["question"]] + " ?")
                )

                # current round
                gt_option_ind = utterance["gt_index"]
                option_inds = []
                option_inds.append(gt_option_ind)  # [gt_index ]
                all_inds = list(range(100))
                all_inds.remove(gt_option_ind)
                all_inds = all_inds[: (num_options - 1)]
                option_inds.extend(all_inds)  # [gt_index, 0~99]
                gt_option_inds.append(0)  # [0]
                cur_rnd_options = []
                answer_options = [utterance["answer_options"][k] for k in option_inds]
                assert len(answer_options) == len(option_inds) == num_options
                assert answer_options[0] == utterance["answer"]

                if rnd == self.visdial_data_val_dense[index]["round_id"] - 1:
                    gt_relevance = torch.Tensor(
                        self.visdial_data_val_dense[index]["gt_relevance"]
                    )
                    gt_relevance = gt_relevance[torch.LongTensor(option_inds)]

                for answer_option in answer_options:
                    cur_rnd_cur_option = cur_rnd_utterance.copy()
                    cur_rnd_cur_option.append(
                        self.tokenizer.encode(cur_answers[answer_option])
                    )
                    cur_rnd_options.append(cur_rnd_cur_option)
                cur_rnd_utterance.append(
                    self.tokenizer.encode(cur_answers[utterance["answer"]])
                )
                utterances.append(cur_rnd_utterance)
                options_all.append(cur_rnd_options)

            tokens_all = []
            padding_mask_all = []
            segments_all = []
            masked_all = []

            for rnd, cur_rnd_options in enumerate(options_all):
                tokens_all_rnd = []
                padding_mask_all_rnd = []
                segments_all_rnd = []
                masked_all_rnd = []

                for i, cur_rnd_option in enumerate(cur_rnd_options):
                    cur_rnd_option, _ = pruneRounds(
                        cur_rnd_option, self.params["visdial_tot_rounds"]
                    )
                    tokens, segments, mask, padding_mask = self.encode_input(
                        cur_rnd_option,
                        self.MASK,
                        self.CLS,
                        self.SEP,
                        max_seq_len=MAX_SEQ_LEN,
                        mask_prob=0,
                    )

                    tokens_all_rnd.append(tokens)
                    padding_mask_all_rnd.append(padding_mask)
                    segments_all_rnd.append(segments)
                    masked_all_rnd.append(mask)

                tokens_all.append(torch.cat(tokens_all_rnd, 0).unsqueeze(0))
                padding_mask_all.append(torch.cat(padding_mask_all_rnd, 0).unsqueeze(0))
                segments_all.append(torch.cat(segments_all_rnd, 0).unsqueeze(0))
                masked_all.append(torch.cat(masked_all_rnd, 0).unsqueeze(0))

            tokens_all = torch.cat(tokens_all, 0)
            padding_mask_all = torch.cat(padding_mask_all, 0)
            segments_all = torch.cat(segments_all, 0)
            masked_all = torch.cat(masked_all, 0)

            item = {}
            item["tokens"] = tokens_all
            item["padding_mask"] = padding_mask_all
            item["segments"] = segments_all
            item["gt_option_inds"] = torch.LongTensor(gt_option_inds)
            item["round_id"] = torch.LongTensor(
                [self.visdial_data_val_dense[index]["round_id"]]
            )
            item["gt_relevance"] = gt_relevance
            item["mask"] = masked_all

            features, num_boxes, boxes, _, image_target = self._image_features_reader[
                img_id
            ]
            features, img_loc, image_label, image_target, image_mask = self.encode_img(
                features, num_boxes, boxes, image_target, mask_prob=0
            )

            item["image_feat"] = features
            item["image_loc"] = img_loc
            item["image_id"] = torch.LongTensor([img_id])
            item["image_label"] = image_label
            item["image_target"] = image_target
            item["image_mask"] = image_mask
            return item
        else:
            assert num_options == 100
            cur_rnd_utterance = [self.tokenizer.encode(dialog["caption"])]
            options_all = []
            for rnd, utterance in enumerate(dialog["dialog"]):
                cur_rnd_utterance.append(
                    self.tokenizer.encode(cur_questions[utterance["question"]])
                )
                if rnd != len(dialog["dialog"]) - 1:
                    cur_rnd_utterance.append(
                        self.tokenizer.encode(cur_answers[utterance["answer"]])
                    )
            for answer_option in dialog["dialog"][-1]["answer_options"]:
                cur_option = cur_rnd_utterance.copy()
                cur_option.append(self.tokenizer.encode(cur_answers[answer_option]))
                options_all.append(cur_option)

            tokens_all = []
            mask_all = []
            segments_all = []
            padding_mask_all = []

            for j, option in enumerate(options_all):
                option, _ = pruneRounds(option, self.params["visdial_tot_rounds"])
                # print("option: {}{}".format(j,tokens2str(option)))
                tokens, segments, mask, padding_mask = self.encode_input(
                    option,
                    self.MASK,
                    self.CLS,
                    self.SEP,
                    max_seq_len=MAX_SEQ_LEN,
                    mask_prob=0,
                )

                tokens_all.append(tokens)
                mask_all.append(mask)
                segments_all.append(segments)
                padding_mask_all.append(padding_mask)

            tokens_all = torch.cat(tokens_all, 0)
            mask_all = torch.cat(mask_all, 0)
            segments_all = torch.cat(segments_all, 0)
            padding_mask_all = torch.cat(padding_mask_all, 0)

            item = {}
            item["tokens"] = tokens_all.unsqueeze(0)
            item["segments"] = segments_all.unsqueeze(0)
            item["padding_mask"] = padding_mask_all.unsqueeze(0)
            item["mask"] = mask_all.unsqueeze(0)

            item["image_id"] = torch.LongTensor([img_id])
            item["round_id"] = torch.LongTensor([dialog["round_id"]])

            # add image features. Expand them to create batch * num_rounds * num options * num bbox * img feats
            features, num_boxes, boxes, _, image_target = self._image_features_reader[
                img_id
            ]
            features, spatials, image_mask, image_target, image_label = self.encode_img(
                features, num_boxes, boxes, image_target, max_regions=37, mask_prob=0
            )

            item["image_feat"] = features
            item["image_loc"] = spatials
            item["image_mask"] = image_mask
            item["image_target"] = image_target
            item["image_label"] = image_label

            return item