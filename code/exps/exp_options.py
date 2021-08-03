import argparse

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', type=int, help='dataset id')
    parser.add_argument('--model_path', help='path to the model')
    parser.add_argument('--cos_thres_list', help='theshold list for the cosine similarity')
    parser.add_argument('--model_arch', type=int, default=0)
    parser.add_argument('--pretrained_autoenc_model_path')
    parser.add_argument('--data_dir', help='training data dir')
    parser.add_argument('--min_word_freq', type=int, help='min word freq')
    parser.add_argument('--batch_size', type=int, help='batch size which was used in training')
    parser.add_argument('--reverse_src', action='store_true')
    parser.add_argument('--drate', type=float, help='dropout rate')
    parser.add_argument('--sel_cols')

    parser.add_argument('--slice_size', type=int)
    parser.add_argument('--slice_size_A', type=int)
    parser.add_argument('--slice_size_B', type=int)

    parser.add_argument('--sif', action='store_true')
    parser.add_argument('--sif_param', type=float)

    parser.add_argument('--lsh_config')
    parser.add_argument('--topk_config')

    parser.add_argument('--gpu', action='store_true')

    parser.add_argument('--rnn_layers', type=int)
    parser.add_argument('--rnn_hidden_size', type=int, help='rnn hidden size')


    parser.add_argument('--seq2seq_model', help='path to the seq2seq model')
    parser.add_argument('--cls_model', help='path to the auxiliary classifier model')

    parser.add_argument('--structured', action='store_true')
    parser.add_argument('--textual', action='store_true')
    parser.add_argument('--textual_with_title', action='store_true')

    parser.add_argument('--output_candset', action='store_true')
    parser.add_argument('--output_candset_path')

    # for seq2seq
    parser.add_argument('--train_data_dir')
    parser.add_argument('--table_data_dir')

    parser.add_argument('--word_list')
    parser.add_argument('--word_emb')

    parser.add_argument('--single_table', action='store_true')

    parser.add_argument('--pca_dim', type=int)

    parser.add_argument('--rm_pc', action='store_true')

    return parser
