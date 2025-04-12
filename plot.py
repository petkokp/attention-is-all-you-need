import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

PATH = 'experiments/en-de/04_12_2025/12_40_09' # TODO - path add as an argument to the script, not hardcoding like that
NUM_EPOCHS = 100 # TODO get this from the config file

def plot_losses():
    train_csv_path = os.path.join(PATH, 'scalars', 'loss', 'train.csv')
    valid_csv_path = os.path.join(PATH, 'scalars', 'loss', 'validation.csv')
    if not os.path.exists(train_csv_path) or not os.path.exists(valid_csv_path):
        print(f"Warning: Loss CSV files not found in {PATH}. Skipping loss plot.")
        return
    try:
        train_df = pd.read_csv(train_csv_path, header=None).head(NUM_EPOCHS)
        valid_df = pd.read_csv(valid_csv_path, header=None).head(NUM_EPOCHS)

        sns.lineplot(data=train_df, x=0, y=1, label='Train Loss') # Use data=df syntax
        ax = sns.lineplot(data=valid_df, x=0, y=1, label='Validation Loss')
        ax.set(xlabel='Epoch', ylabel='Loss', title='Training and Validation Losses')
        plt.legend() # Add legend
        plt.grid(True, linestyle='--', alpha=0.6) # Add grid
        plt.tight_layout() # Adjust layout
        plt.savefig(os.path.join(PATH, 'losses.png'))
        plt.show()
        plt.close() # Close the figure
    except Exception as e:
        print(f"Error plotting losses: {e}")


def plot_lr():
    lr_csv_path = os.path.join(PATH, 'scalars', 'lr.csv')
    if not os.path.exists(lr_csv_path):
        print(f"Warning: LR CSV file not found in {PATH}. Skipping LR plot.")
        return
    try:
        lr_df = pd.read_csv(lr_csv_path, header=None).head(NUM_EPOCHS)

        ax = sns.lineplot(data=lr_df, x=0, y=1) # Use data=df syntax
        ax.set(xlabel='Epoch', ylabel='Learning Rate',
               title='Learning Rate Schedule')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(PATH, 'lr.png'))
        plt.show()
        plt.close() # Close the figure
    except Exception as e:
        print(f"Error plotting learning rate: {e}")


def plot_bleu():
    bleu_csv_path = os.path.join(PATH, 'scalars', 'bleu.csv')
    if not os.path.exists(bleu_csv_path):
        print(f"Warning: BLEU CSV file not found in {PATH}. Skipping BLEU plot.")
        return
    try:
        bleu_df = pd.read_csv(bleu_csv_path, header=None).head(NUM_EPOCHS)
        # Ensure column 1 exists and is numeric before multiplying
        if 1 in bleu_df.columns:
             bleu_df[1] = pd.to_numeric(bleu_df[1], errors='coerce') * 100
             bleu_df = bleu_df.dropna(subset=[1]) # Drop rows where conversion failed
        else:
             print("Warning: Column 1 (BLEU score) not found in bleu.csv")
             return


        ax = sns.lineplot(data=bleu_df, x=0, y=1) # Use data=df syntax
        ax.set(xlabel='Epoch', ylabel='BLEU Score (%)', # Indicate percentage
               title='Validation BLEU Score')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(bottom=0) # BLEU score cannot be negative
        plt.tight_layout()
        plt.savefig(os.path.join(PATH, 'bleu.png'))
        plt.show()
        plt.close() # Close the figure
    except Exception as e:
        print(f"Error plotting BLEU score: {e}")


if __name__ == '__main__':
    print(f"Plotting results from: {PATH}")
    print(f"Plotting up to epoch: {NUM_EPOCHS}")

    plot_losses()
    plot_lr()
    plot_bleu()
