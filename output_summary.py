import pandas as pd


def compare(df, c1, c2):
    return sum(df[c1] == df[c2]) / len(df)


def summary(output):
    output = pd.read_csv(output, index_col=0)
    g1 = output[(output['labels'] == 0) & (output['aux_labels'] == 0)]
    g2 = output[(output['labels'] == 0) & (output['aux_labels'] == 1)]  #
    g3 = output[(output['labels'] == 1) & (output['aux_labels'] == 0)]  #
    g4 = output[(output['labels'] == 1) & (output['aux_labels'] == 1)]
    
    for i, g in enumerate([output, g1, g2, g3, g4]):
        print(f"=====Group {i} {(len(g))}=====")
        print("Prediction accuracy:", compare(g, 'labels', 'predictions'))
        for simfunc in ["dotprod", "cossim", "l2sim"]:
            for a in ['labels', 'predictions', 'aux_labels']:
                for b in ['train_rank_top_label_' + simfunc, 'train_rank_top_aux_label_' + simfunc]:
                    print(f"Compare {a} and {b}", compare(g, a, b))


if __name__ == "__main__":
    # # summary("output_new.csv")
    # # summary("output_grad_new.csv")
    # output = pd.read_csv("output_grad_new.csv", index_col=0)
    # # predict_wrong_0 = output[(output['labels'] == 1) & (output['train_rank_top_label_dotprod'] == 0)]
    # # spu_prop = sum(predict_wrong_0['aux_labels'] == 0)/len(predict_wrong_0)
    # # print(spu_prop)
    
    # # predict_wrong_0_ = output[(output['labels'] == 1) & (output['predictions'] == 0)]
    # # spu_prop_ = sum(predict_wrong_0_['aux_labels'] == 0)/len(predict_wrong_0_)
    # # print(spu_prop_)

    # # tmp = output[(output['labels'] == 0) & (output['predictions'] == 0) & (output['aux_labels'] == 1)]
    # # print(sum(tmp['train_rank_top_label_l2sim'] == 1) / len(tmp))

    # label_pred_dif = output[(output['labels'] != output['predictions'])]
    # label_pred_dif_minority = label_pred_dif[(label_pred_dif['labels'] != label_pred_dif['aux_labels'])]
    # print(len(label_pred_dif_minority) / len(label_pred_dif))

    # label_pred_dif = output[(output['labels'] != output['train_rank_top_label_dotprod'])]
    # label_pred_dif_minority = label_pred_dif[(label_pred_dif['labels'] != label_pred_dif['aux_labels'])]
    # print(len(label_pred_dif_minority) / len(label_pred_dif))

    # minority = output[(output['labels'] != output['aux_labels'])]
    # predict_wrong = minority[(minority['labels'] != minority['predictions'])]
    # rank_wrong = minority[(minority['labels'] != minority['train_rank_top_label_dotprod'])]
    # print(len(predict_wrong) / len(minority))
    # print(len(rank_wrong) / len(minority))
    
    # output = pd.read_csv("output_grad_loss_eval_try_2.csv", index_col=0)
    # g1 = output[(output['labels'] == 0) & (output['aux_labels'] == 0)]
    # g2 = output[(output['labels'] == 0) & (output['aux_labels'] == 1)]  #
    # g3 = output[(output['labels'] == 1) & (output['aux_labels'] == 0)]  #
    # g4 = output[(output['labels'] == 1) & (output['aux_labels'] == 1)]

    # for i, g in enumerate([output, g1, g2, g3, g4]):
    #     print(f"=====Group {i} {(len(g))}=====")
    #     print(g['loss'].mean())

    output = pd.read_csv("output_test_grad_loss_feat.csv", index_col=0)
    g1 = output[(output['labels'] == 0) & (output['aux_labels'] == 0)]
    g2 = output[(output['labels'] == 0) & (output['aux_labels'] == 1)]  #
    g3 = output[(output['labels'] == 1) & (output['aux_labels'] == 0)]  #
    g4 = output[(output['labels'] == 1) & (output['aux_labels'] == 1)]

    for col in ["loss", "feat_norm", "grad_norm"]:
        for i, g in enumerate([output, g1, g2, g3, g4]):
            print(f"========== {col} ==========")
            print(f"=====Group {i} {(len(g))}=====")
            print(g[col].mean())
    
    print("========== ACC ==========")
    for i, g in enumerate([output, g1, g2, g3, g4]):
        print(f"=====Group {i} {(len(g))}=====")
        print((g["labels"] == g["predictions"]).mean())