import argparse
# ===-----------------------------------------------------------------------===
# Argument parsing
# ===-----------------------------------------------------------------------===
def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='is training?')
    parser.add_argument('--gpu', action='store_true', help='use gpu?')
    parser.add_argument('--lrate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--drate', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of full passes through training set')
    parser.add_argument('--batch_size', default=128, type=int, help='batch_size')

    parser.add_argument('--lrdecay', default=1.0, type=float, help='learning rate decay value')
    parser.add_argument('--grad_clip_thrsd', default=5.0, type=float, help='Gradient clipping threshold. If 0, not uses it.')
    parser.add_argument('--sparse_emb', default=False, action='store_true', help='Use sparse embeddings for word and character vectors?')
    parser.add_argument('--l2reg_coef', default=0, type=float, help='L2 regularization coefficient')
    parser.add_argument('--weight_decay', default=0, type=float, help='Weight decay of the entire model')
    parser.add_argument('--seed', default=1000, type=int, help='seed for random')
    parser.add_argument('--max_grad_norm', help='If the norm of the gradient vector exceeds this renormalize it\
                               to have the norm equal to max_grad_norm', type=float, default=5)

    parser.add_argument('--model_arch', default=0, type=int, help='0: Vanilla autoencoder.')
    parser.add_argument('--model', dest='model_path', help='Model directory path', required=True)
    parser.add_argument('--data', help='Training data file path', required=True)
    parser.add_argument('--test', help='Test data file path')
    parser.add_argument('--single_table', action='store_true', help='only one input table?')
    parser.add_argument('--sel_cols', help='selected columns in the schema')
    parser.add_argument('--word_list', required=True)
    parser.add_argument('--word_emb', required=True)

    parser.add_argument('--deterministic', action='store_true', help='produce deterministic results by fixing the random seed.')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--concat', action='store_true')
    parser.add_argument('--large_dataset', action='store_true')
    parser.add_argument('--pos_weight', type=float)
    parser.add_argument('--multi_class_weight')

    # For SIF.
    parser.add_argument('--sif', action='store_true')
    parser.add_argument('--sif_param', type=float)
    parser.add_argument('--word_freq_file')

    # For autoencoder training.
    parser.add_argument('--encoder_dims')
    parser.add_argument('--decoder_dims')
    parser.add_argument('--denoise', action='store_true')
    parser.add_argument('--noise_rate', default=0.0, type=float, help='noise rate for denoising encoder')

    parser.add_argument('--autoenc_model')

    # For self-teaching training.
    parser.add_argument('--prime_enc_dims')
    parser.add_argument('--aux_enc_dims')
    parser.add_argument('--cls_enc_dims')

    # For seq2seq training.
    parser.add_argument('--min_word_freq', type=int)
    parser.add_argument('--rnn_layers', type=int)
    parser.add_argument('--reverse_src', action='store_true')
    parser.add_argument('--rnn_hidden_size', type=int, help='hidden size of rnn in seq2seq')
    parser.add_argument('--teacher_forcing_ratio', type=float, help='forcing ratio of rnn decoding')

    parser.add_argument('--data_seq2seq')
    parser.add_argument('--seq2seq_model')

    parser.add_argument('--rm_pc', action='store_true')

    return parser
