import numpy as np
from PIL import Image
from tqdm import tqdm

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import argparse
from glob import glob


def draw(data):
    print(len(data['images']))
    print(data['images'][0])
    fig = make_subplots(rows=len(data['images']), cols=4, specs=[
        [{'type': 'image'}, {'type': 'xy'}, {'type': 'xy'}, {'type': 'table'}]
    ] * len(data['images']),
    subplot_titles=("image", "AD", "HP", ""))
    for idx, (img, probs, gt) in tqdm(enumerate(zip(data['images'], data['probs'], data['gts']))):
        idx = idx + 1
        img = Image.open(img)
        fig.add_trace(
            go.Image(z=img),
            row=idx, col=1
        )

        fig.add_trace(
            go.Histogram(x=probs[:, 0]),
            row=idx, col=2
        )

        fig.add_trace(
            go.Histogram(x=probs[:, 1]),
            row=idx, col=3
        )

        fig.add_trace(
            go.Table(header=dict(values=['statistics', '0', '1']),
                    cells=dict(
                        values=[
                            ['min', 'max', 'median', 'average', 'std', 'entropy'], 
                            [round(min(probs[:, 0])), round(max(probs[:, 0])), round(np.median(probs[:, 0])), round(np.mean(probs[:, 0])), round(np.std(probs[:, 0])), '-'],
                            [round(min(probs[:, 1])), round(max(probs[:, 1])), round(np.median(probs[:, 1])), round(np.mean(probs[:, 1])), round(np.std(probs[:, 1])), '-'],
                        ])),
            row=idx, col=4
        )
    fig.update_layout(height=500 * len(data['images']), width=1800, title_text="Stacked Subplots")
    return fig


def main():
    parser = argparse.ArgumentParser("viz arg parser", add_help=False)
    parser.add_argument("--output", help="viz output path")
    args = parser.parse_args()

    mc_dropout_results = glob(f"{args.output}/*.npy")

    for result in mc_dropout_results:
        data = np.load(result, allow_pickle=True).item()
        fig = draw(data)
        fig.write_html(result.replace('npy', 'html'))


if __name__ == "__main__":
    main()
