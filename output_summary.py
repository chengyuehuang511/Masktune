import pandas as pd


def compare(df, c1, c2):
    return sum(df[c1] == df[c2]) / len(df)

if __name__ == "__main__":
    output = pd.read_csv("output.csv", index_col=0)
    g1 = output[(output['labels'] == 0) & (output['aux_labels'] == 0)]
    g2 = output[(output['labels'] == 0) & (output['aux_labels'] == 1)]  #
    g3 = output[(output['labels'] == 1) & (output['aux_labels'] == 0)]  #
    g4 = output[(output['labels'] == 1) & (output['aux_labels'] == 1)]

    print(output)
    
    for i, g in enumerate([output, g1, g2, g3, g4]):
        print(f"=====Group {i}=====")
        print("Prediction accuracy:", compare(g, 'labels', 'predictions'))
        for simfunc in ["dotprod", "cossim", "l2sim"]:
            for a in ['labels', 'predictions', 'aux_labels']:
                for b in ['train_rank_top_label_' + simfunc, 'train_rank_top_aux_label_' + simfunc]:
                    print(f"Compare {a} and {b}", compare(g, a, b))
    