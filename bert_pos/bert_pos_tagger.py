import nltk
import os
import collections

# pos tag dictionary:
#  MASK-0  CC-1   CD-2   DT-3   EX-4   FW-5   IN-6   JJ-7
# JJR-8  JJS-9  LS-10   MD-11  NN-12  PDT-13  POS-14
# PRP-15  RB-16  RBR-17  RBS-18  RP-19  TP-20  UH-21
# VB-22 VBD-23 VBG-24 VBN-25 VBP-26 VBZ-27 WDT-28 WP-29 WP$-30
# WRB-31 NNS-12 NNP-12 NNPS-12 PRP$-15 UNKNOWN-32


class PosSentTagger(object):
    def __init__(self, pos_tag_file):
        if not os.path.exists(pos_tag_file):
            raise ValueError(
                "Can't find a pos_tag file at path {}".format(pos_tag_file)
            )

        self.tag_dic = self.load_post_tag(pos_tag_file)
        self.id_to_tag = collections.OrderedDict(
            [(ids, tag) for tag, ids in self.tag_dic.items()]
        )
        self.unk_tag = "unknown"

    def load_post_tag(self, file):
        tag_dic = collections.OrderedDict()
        tag_dic["mask"] = 0
        with open(file, "r") as reader:
            tags = reader.readlines()
        i = 1
        for _, tag in enumerate(tags):
            tag = tag.rstrip("\n")
            if tag == "NNS" or tag == "NNP" or tag == "NNPS" or tag == "PRP$":
                continue
            tag_dic[tag] = i
            i += 1
        tag_dic["NNS"] = tag_dic["NN"]
        tag_dic["NNP"] = tag_dic["NN"]
        tag_dic["NNPS"] = tag_dic["NN"]
        tag_dic["PRP$"] = tag_dic["PRP"]
        tag_dic["unknown"] = i
        return tag_dic

    @property
    def tag_size(self):
        return len(set(self.tag_dic.values()))

    def _convert_tag_to_id(self, tag):
        """Converts a tag in an id using tag_dic"""
        return self.tag_dic.get(tag, self.tag_dic.get(self.unk_tag))

    def _convert_id_to_tag(self, index):
        """Converts an index in a tag"""
        return self.id_to_tag.get(index, self.unk_tag)

    def _convert_tags_to_ids(self, tags):
        """Convets a tag list to id list"""
        tag_list = []
        for tag in tags:
            tag_list.append(self._convert_tag_to_id(tag))
        return tag_list

    def _tagger(self, word_list):
        tag_list = []
        for i, (word, pos) in enumerate(nltk.tag.pos_tag(word_list)):
            if word in ["[SEP]", "[MASK]", "[CLS]", "[PAD]"]:
                tag_list.append("unknown")
            elif "#" in word:
                if len(tag_list) > 0:
                    tag = tag_list[-1]
                    tag_list.append(tag)
                else:
                    tag_list.append(pos)
            else:
                tag_list.append(pos)
        return tag_list

    def encode(self, word_list):
        tags = self._tagger(word_list)
        ids = self._convert_tags_to_ids(tags)
        return ids

    def _replace_pro_to_noun(self, word_list):
        tags = self._tagger(word_list)
        for i, tag in enumerate(tags):
            if tag == "PRP" or tag == "PRP$" and word_list[i] != "you":
                tags[i] = "NN"
        return tags

    def get_target(self, word_list):
        tags = self._replace_pro_to_noun(word_list)
        ids = self._convert_tags_to_ids(tags)
        return ids

    def detec_pron(self, word_list):
        tag_list = []
        tags = self._tagger(word_list)
        for i, tag in enumerate(tags):
            if (tag == "PRP" or tag == "PRP$") and word_list[i] != "you":
                tag_list.append(1)
            else:
                tag_list.append(0)
        return tag_list


if __name__ == "__main__":
    word_list = [
        "[CLS]",
        "the",
        "large",
        "clay",
        "[SEP]",
        "jar",
        "is",
        "beside",
        "a",
        "staircase",
        "[MASK]",
        "can",
        "you",
        "tell",
        "if",
        "there",
        "is",
        "anything",
        "inside",
        "the",
        "jar",
        "?",
        "[MASK]",
        "i",
        "can",
        "'",
        "t",
        "tell",
        "[PAD]",
    ]
    t = PosSentTagger("pos_tag_dictionary")
    print(t.tag_dic)
    print(t._tagger(word_list))
    print(t.encode(word_list))
    print(t._tagger(["fire", "##fighter"]))
    # print(t.get_target(word_list))
    # tag = t._tagger(['he', 'she', 'it', 'their', '.', 'you', 'are', 'there', 'people', 'on', 'the', 'sidewalks'])
    # print(t.detec_pron(['he', 'she', 'it', 'their', '.', 'you', 'are', 'there', 'people', 'on', 'the', 'sidewalks']))
    # print(tag)
    # print(t._replace_pro_to_noun(tag))
    print(
        t.encode(
            [
                "he",
                "she",
                "it",
                "their",
                ".",
                "you",
                "are",
                "there",
                "people",
                "on",
                "the",
                "sidewalks",
            ]
        )
    )
    print(
        t.get_target(
            [
                "he",
                "she",
                "it",
                "their",
                ".",
                "you",
                "are",
                "there",
                "people",
                "on",
                "the",
                "sidewalks",
            ]
        )
    )
    print(
        t.detec_pron(
            [
                "he",
                "she",
                "it",
                "their",
                ".",
                "you",
                "are",
                "there",
                "people",
                "on",
                "the",
                "sidewalks",
            ]
        )
    )
    # print(t._convert_tags_to_ids(['PRP', 'PRP', 'PRP', 'PRP#', '.', 'PRP', 'VBP', 'RB', 'NNS', 'IN', 'DT', 'NNS']))
