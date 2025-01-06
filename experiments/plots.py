import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

df = pd.read_csv("../results/test.csv")
i = 0
for group in df['f_0'].unique():
    subset = df[df['f_0'] == group]  
    # group = group.replace("(", r"\text{")
    # group = group.replace(")", r"}")
    # group = group.replace("*", r"\cdot")
    # group = group.replace(" ^ ", r"^")

    plt.figure(figsize=(8, 5))
    plt.plot(subset['m_val'], subset['eo_val_0'], label="EO Value for Y = 0", marker='o', linestyle='-')
    plt.plot(subset['m_val'], subset['eo_val_1'], label="EO Value for Y = 1", marker='o', linestyle='-')

    plt.xlabel('Number of Samples')
    plt.ylabel('Equalized Odds Value')
    plt.title(f'Function: {group}')
    plt.legend(loc="best", fontsize=12)
    plt.savefig(f'../figures/plot_{str(i)}.png')
    plt.show()
    
    i += 1
    