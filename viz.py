from curses import KEY_REPLACE
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import argparse
from glob import glob
import csv
from shutil import copyfile

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import json



def chunks(xs, n):
    n = max(1, n)
    return (xs[i:i+n] for i in range(0, len(xs), n))


def draw_subplots(img, gt_name, pred, p, name, idx):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img)
    axes[0].set_title(gt_name)

    # if pred == 0:
    sns.histplot(data=p[:, 0], kde=True, ax=axes[1])
    # axes[1].set_title(f"AD")
    # print(name + f"/{idx}.png")
    fig.tight_layout()
    fig.savefig(name + f"/{idx}.png")
    plt.close(fig)
    # axes[0].clear()
    # axes[1].clear()


def draw(data, name):
    f = open(name + "/results.csv", "w")
    wr = csv.writer(f)
    wr.writerow(['img_num'] + [i for i in range(100)])

    origin_f = open(name + "/origin_name.csv", "w")
    origin_wr = csv.writer(origin_f)
    origin_wr.writerow(['origin_name', 'img_num'])

    if not os.path.exists(name + "/images"):
        os.makedirs(name + "/images")

    # images are image name list
    # probs are probability list of each image
    # shape [num_images, num_samples, 2]
    # num_samples mean mc dropout samples
    # 2 means 2 classes 0 is AD, 1 is HP
    # gts are ground truth list of each image
    for idx, (i, p, g) in enumerate(zip(data['images'], data['probs'], data['gts'])):
        mean = p.mean(axis=0)
        ad = mean[0]
        hp = mean[1]
        ad_std = p.std(axis=0)
        ad_std = [ad - ad_std[0], ad + ad_std[0]]
        hp_std = p.std(axis=0)
        hp_std = [hp - hp_std[1], hp + hp_std[1]]
        # print(ad)
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            # delta = {'reference': 0},
            value = ad if ad > hp else hp,
            number = {'valueformat': '.0%'},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {
                    'range': [None, 1], 
                    'tickmode': 'array',
                    'nticks':11,
                    'tickvals': [i * 0.1 for i in range(11)],
                    'tickformat': '.0%'
                },
                'bar': {'color': "blue" if ad > hp else "red"},
                'steps' : [
                    {'range': ad_std if ad > hp else hp_std, 'color': 'gray'},],
                # 'threshold' : {
                #     'line': {'color': "red", 'width': 4},
                #     'thickness': 0.75,
                #     'value': 90}
            }
        ))
        fig.update_layout(
            annotations=[
                dict(
                    x=0.5,
                    y=0.3,
                    text="Hyperplastic Polyp" if ad < hp else "Adenomatous Polyp",
                    showarrow=False,
                    font=dict(size=35, color="black"),
                    align="center",
                    bgcolor="white",
                    xref="paper",
                    yref="paper"
                )
            ]
        )
        # fig.tight_layout()
        fig.write_image(name + f"/{idx}.png")

        # left = Image.open(i)
        # right = Image.open(name + f"/{idx}.png")
        # left.thumbnail((left.width,left.height))
        # resize left image to match right image with aspect ratio


        # concat
        # concat = Image.new('RGB', (left.width + right.width, left.height))

        # plt.close(fig)
        
        # wr.writerow([idx] + p[:, 0].tolist())
        # origin_wr.writerow([i.split('/')[-1], idx])
        # pred = p.mean(axis=0).argmax(axis=0)
        # copyfile(i, name + "/images/" + str(idx) + ".png")
        # i = Image.open(i)
        # draw_subplots(i, "AD" if g == 0 else "HP", pred, p, name, idx)

    f.close()
        # break
        #     fig.add_trace(
        #         go.Image(z=i, name="AD" if g == 0 else "HP"),
        #         row=idx, col=1
        #     )

        #     # fig_dist = ff.create_distplot([p[:, 0], p[:, 1]], ['AD', 'HP'], bin_size=.001)
        #     # print(p[:, 0].shape)
        #     if pred == 0:
        #         # print(fig_dist['data'][0])
        #         fig.add_trace(
        #             go.Histogram(
        #                 x = p[:, 0], 
        #                 marker_color='blue', nbinsx=50),
        #             row=idx, col=2
        #         )

        #         fig.add_shape(type="line",x0=np.mean(p[:, 0]), x1=np.mean(p[:, 0]), y0 =0, y1=12 , xref='x', yref='y',
        #         line = dict(color = 'green', dash = 'dash'), row=idx, col=2)
        #         fig.add_trace(go.Scatter(x=[np.mean(p[:, 0])], y=[12], name = 'mean',
        #                      showlegend=False,
        #                      mode='markers', marker=dict(color = 'green', size=6)), row=idx, col=2)
        #     else:
        #         fig.add_trace(
        #             go.Histogram(
        #                 x = p[:, 1], 
        #                 marker_color='red', nbinsx=50),
        #             row=idx, col=2
        #         )

        #         # fig.add_trace(
        #         #     go.Scatter(
        #         #         fig_dist['data'][3],
        #         #         line=dict(color='red', width=0.5)),
        #         #     row=idx, col=2
        #         # )
                
        #         fig.add_shape(type="line",x0=np.mean(p[:, 1]), x1=np.mean(p[:, 1]), y0 =0, y1=12 , xref='x', yref='y',
        #         line = dict(color = 'green', dash = 'dash'), row=idx, col=2)
        #         fig.add_trace(go.Scatter(x=[np.mean(p[:, 1])], y=[12], name = 'mean',
        #                      showlegend=False,
        #                      mode='markers', marker=dict(color = 'green', size=6)), row=idx, col=2)

        #     # print(round(min(probs[:, 0]), 3))
        #     fig.add_trace(
        #         go.Table(header=dict(values=['statistics', 'AD', 'HP']),
        #                 cells=dict(
        #                     values=[
        #                         ['min', 'max', 'median', 'mean', 'std', 'entropy'], 
        #                         [str(round(min(p[:, 0]), 3)), 
        #                         str(round(max(p[:, 0]), 3)), 
        #                         str(round(np.median(p[:, 0]), 3)), 
        #                         str(round(np.mean(p[:, 0]), 3)), 
        #                         str(round(np.std(p[:, 0]), 3)), '-'],
        #                         [
        #                             str(round(min(p[:, 1]), 3)), 
        #                             str(round(max(p[:, 1]), 3)), 
        #                             str(round(np.median(p[:, 1]), 3)), 
        #                             str(round(np.mean(p[:, 1]), 3)), 
        #                             str(round(np.std(p[:, 1]), 3)), '-'],
        #                     ])),
        #         row=idx, col=3
        #     )
        # fig.update_layout(height=500 * len(imgs), width=450*3, title_text="Stacked Subplots")
        # fig.write_html(name + f"/{chunk_idx}.html")


def main():
    parser = argparse.ArgumentParser("viz arg parser", add_help=False)
    parser.add_argument("--output", help="viz output path")
    args = parser.parse_args()

    mc_dropout_results = glob(f"{args.output}/*.npy")

    for result in mc_dropout_results:
        if not os.path.exists(result[:-4]):
            os.makedirs(result[:-4])
        data = np.load(result, allow_pickle=True).item()
        # print(data['probs'].tolist())
        # print(data['gts'].shape)

        # save as json
        # with open(f"{args.output}.json", "w") as f:
        #     data = {
        #         "probs": data['probs'].tolist(),
        #         "gts": data['gts'].tolist(),
        #         "images": data['images']
        #     }
        #     json.dump(data, f)
        draw(data, result[:-4])
        # fig


if __name__ == "__main__":
    main()
