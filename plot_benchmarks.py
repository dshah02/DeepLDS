import json
import matplotlib.pyplot as plt
import argparse
import os
import yaml
import numpy as np


def curve_L(L, a, b):
    return a * (L) + b

def curve(L, a, b):
    return a * (L**2 * np.log(L + 1e-9)) + b

def curve_Llogl(L, a, b):
    return a * (L * np.log(L + 1e-9)) + b

def curve_L2(L, a, b):
    return a * (L**2) + b

def main(path_to_outputs:str, display:bool = False):
    with open(path_to_outputs, "r") as file:
        data = json.load(file)

    x, y = list(map(int, data.keys())), list(data.values())

    parent_dir = os.path.dirname(path_to_outputs)
    eval_config = yaml.load(open(os.path.join(parent_dir, "eval_config.yaml"), 'r'), Loader=yaml.FullLoader)
    model_eval_config = yaml.load(open(os.path.join(parent_dir, "model_config.yaml"), 'r'), Loader=yaml.FullLoader)

    # filename follows format -> runname=<gen>-max_input_seqlen=<num>-max_new_output_seqlen=all.json
    parts = path_to_outputs.split('-')
    runname = parts[0].split('=')[1]
    max_input_seq = int(parts[1].split('=')[1])

    # L fit
    X = np.array(x)
    A = np.column_stack([X, np.ones_like(X)])   
    coeffs, _, _, _ = np.linalg.lstsq(A, np.array(y), rcond=None)
    a_fit, b_fit = coeffs

    # L**2 fit
    X = np.array(x)**2
    A = np.column_stack([X, np.ones_like(X)])   
    coeffs, _, _, _ = np.linalg.lstsq(A, np.array(y), rcond=None)
    a_fit_bis_bis, b_fit_bis_bis = coeffs

    # # L**2 log(L) fit
    # X = (np.array(x))**2 #* np.log(np.array(x) + 1e-9)
    # A = np.column_stack([X, np.ones_like(X)])   
    # coeffs, _, _, _ = np.linalg.lstsq(A, np.array(y), rcond=None)
    # a_fit, b_fit = coeffs
    # L_plot = np.linspace(min(x), max(x), 200)

    # Llog(L) fit
    X_bis = np.array(x) * np.log(np.array(x) + 1e-9)
    A_bis = np.column_stack([X_bis, np.ones_like(X_bis)])   
    coeffs_bis, _, _, _ = np.linalg.lstsq(A_bis, np.array(y), rcond=None)
    a_fit_bis, b_fit_bis = coeffs_bis
    L_plot = np.linspace(min(x), max(x), 200)

    title = f"Generation runtimes for {runname} decoding on max input seq length of {max_input_seq}"

    plt.figure()
    plt.plot(x, y, marker='o')
    plt.plot(L_plot, curve_L(L_plot, a_fit, b_fit), label=f'Fitted ${a_fit:.2e}\\, L**2 + {b_fit:.2e}$')
    plt.plot(L_plot, curve_L2(L_plot, a_fit_bis_bis, a_fit_bis_bis), label=f'Fitted ${a_fit_bis_bis:.2e}\\, L**2 + {b_fit_bis_bis:.2e}$')
    plt.plot(L_plot, curve_Llogl(L_plot, a_fit_bis, b_fit_bis), label=f'Fitted ${a_fit_bis:.2e}\\, L \\log L + {b_fit_bis:.2e}$')

    plt.title(title)
    plt.xlabel("Sequence length")
    plt.ylabel("mean runtime in sec")
    plt.legend()
    plt.grid(True)
    if display:
        plt.show()
    else:
        plt.savefig(path_to_outputs.replace("json", "png"))

if __name__ =="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_to_outputs",
        type=str,
        default="./outputs/295464/runname=baseline_bfloat16-input_seqlen=0-max_output_seqlen=all_bis.json",
        # "871136/runname=baseline_bfloat16_attn-input_seqlen=0-max_output_seqlen=all.json",
        help="Path to outputs that you want to plot"
    )   
    parser.add_argument(
        "--display",
        type=bool,
        default=False,
        help="Whether you want to display or save the figure"
    ) 
    args = parser.parse_args()

    main(args.path_to_outputs, args.display)