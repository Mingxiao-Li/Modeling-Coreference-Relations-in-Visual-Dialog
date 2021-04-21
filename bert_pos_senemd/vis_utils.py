# -*- coding: utf-8 -*-

"""
visualization toolkit:

- draw_map := attention map plot
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def draw_decision(
    q_id,
    img_id,
    bbox_in_pixel,
    ques,
    ans_gt,
    ans_predict,
    cross_attns,
    additional_info="\n",
    image_path="/esat/jade/tmp/GQA/images",
    out_path="model_decision",
):
    """
    Args:
        q_id (str): question id
        img_id (str/int): image id
        bbox_in_pixel (list of coordinates): bounding boxes
        ques (str): question txt
        ans_gt (str): ground truth answer
        ans_predict (str): predicted answer
        cross_attns (list of 2D array): cross attention maps (we need attn scores from [CLS] -> bbox)
        image_path (str): image path
        out_path (str): output directory
    """
    import os

    os.makedirs(out_path, exist_ok=True)

    cls2bbox = []
    for i, attn in enumerate(cross_attns):
        assert len(attn[0].shape) == 1, "cls2bbox attn score should be 1d vector"
        if i == 0:
            cls2bbox = attn[0]
        else:
            cls2bbox += attn[0]

    cls2bbox /= len(cross_attns)  # average

    import cv2

    img_file = os.path.join(image_path, f"{img_id}.jpg")
    img = cv2.imread(img_file)

    heatmap = np.zeros((img.shape[0], img.shape[1]))
    for i, w in enumerate(cls2bbox):
        box = bbox_in_pixel[i]
        # heatmap[int(box[1]):int(box[3]), int(box[0]):int(box[2])] += w
        # x, y in box is inverse of real coordinate, i.e. x is in dimension 1, y in dimension 0
        for r1 in range(int(box[0]), int(box[2])):
            for c1 in range(int(box[1]), int(box[3])):
                heatmap[c1, r1] = max(heatmap[c1, r1], w)

    heatmap = cv2.normalize(
        heatmap, heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    im_color = cv2.addWeighted(heatmap, 0.5, img, 0.5, 0)

    cv2.imwrite(os.path.join(out_path, f"{q_id}.jpg"), im_color)

    with open(os.path.join(out_path, f"{q_id}.txt"), "w") as out_f:
        text = [
            f"Question ID:  {q_id}\n",
            f"Image ID:     {img_id}\n",
            f"Question:     {ques}\n",
            f"Ground Truth: {ans_gt}\n",
            f"Prediction:   {ans_predict}\n",
            additional_info,
        ]
        out_f.writelines(text)


def draw_attn(
    tokens_x,
    tokens_y,
    attns,
    fig=None,
    ax=None,
    colorbar_label="attention weights",
    cmap="Oranges",
    text_on=False,
):
    """
    Args:
        tokens_x (list): x axis of the attn matrix with length W
        tokens_y (list): y axis of the attn matrix with length H
        attns (2d array - attn matrix): matrix with shape [H, W]
        fig (pyplot.Figure object)
        ax (axes.Axes object): subplot object, created from plt.subplots()
        cmap (colors): Oranges / Reds

    Returns:
        ax (axes.Axes object): subplot object
    """
    font = {  #'family': 'normal',
        #'weight': 'normal',
        "size": 15
    }

    if fig is None:
        fig = plt.figure()

    if ax is None:
        ax = plt.gca()

    # the x axis is the 2nd dimention of attns, idx increase from left to right
    # the y axis is the 1st dimention of attns, idx increase from top to bottom
    im = ax.imshow(attns, interpolation="none", cmap=cmap)
    # The following line is magically align a colorbar fit the plot size
    fig.colorbar(im, fraction=0.046, pad=0.04, ax=ax).set_label(
        label=colorbar_label, size=15
    )

    ax.set_xticklabels([""] + tokens_x, fontdict=font)
    ax.set_yticklabels([""] + tokens_y, fontdict=font)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.tick_params(axis="x", rotation=90)

    if text_on:
        for i in range(attns.shape[0]):
            for j in range(attns.shape[1]):
                text = ax.text(
                    j, i, "%.2f" % (attns[i, j]), ha="center", va="center", color="w"
                )
    return ax


if __name__ == "__main__":
    # Test attention map plot
    print("[INFO] Testing attention map plotter")
    attns_map = np.eye(5)
    attns_map[0, 4] = 1
    token_x = ["[CLS]", "apple", "banana", "zoo", "[SEP]"]
    token_y = ["China", "IS", "So", "Great", "[SEP]"]
    f, axes = plt.subplots(1, 5, figsize=(25, 3))
    for i in range(5):
        axes[i] = draw_attn(token_x, token_y, attns_map, f, axes[i])
    plt.tight_layout()
    # plt.show()
    plt.savefig("img")
    plt.clf()