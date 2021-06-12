import subprocess
import os
import glob
import sacrebleu
import argparse

from datetime import datetime
from sacremoses import MosesTokenizer, MosesDetokenizer
from prpe.prpe6.prpe import apply_prpe
import sentencepiece as spm

NOW = datetime.now()

parser = argparse.ArgumentParser(description='Specify pipeline flags')
parser.add_argument('--segment_type', type=str, default='prpe_bpe', help='prpe_bpe, or none, or prpe, or bpe, or prpe_multi. or unigram')
parser.add_argument('--prpe_multi_runs', type=int, default=5, help='number of iterations for prpe_multi')
parser.add_argument('--model_type', type=str, default='rnn', help='rnn or transformer')
parser.add_argument('--in_lang', type=str, default='qz', help='qz or id')
parser.add_argument('--out_lang', type=str, default='es', help='es or en')
parser.add_argument('--domain', type=str, default='religious', help='dataset folder name')
parser.add_argument('--save_steps', type=int, default=10000, help='saves every x steps')
parser.add_argument('--validate_steps', type=int, default=2000, help='opnenmt validates model every x steps')
parser.add_argument('--train_steps', type=int, default=100000, help='trains model for x steps')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--filter_too_long', type=int, default=-1, help='max token length, -1 for no filtering')

opt = parser.parse_args()

in_lang = opt.in_lang
out_lang = opt.out_lang
domain = opt.domain
validate_steps = opt.validate_steps
train_steps = opt.train_steps

# this naming convention doesnt work on windows
FOLDER = f'-{domain}-{opt.model_type}-{opt.segment_type}:{in_lang}-{out_lang}-{NOW.strftime("%m_%d_%Y_%H_%M_%S")}'

SRC_INPUT = f'data/{domain}/train.{in_lang}.txt'
TGT_INPUT = f'data/{domain}/train.{out_lang}.txt'

SRC_PROCESSED = f'model_opennmt/run' + FOLDER  +f'/processed_train.{in_lang}.txt'
TGT_PROCESSED = f'model_opennmt/run' + FOLDER + f'/processed_train.{out_lang}.txt'

SRC_VLD_PROCESSED = f'model_opennmt/run' + FOLDER  +f'/processed_validate.{in_lang}.txt'
TGT_VLD_PROCESSED = f'model_opennmt/run' + FOLDER + f'/processed_validate.{out_lang}.txt'

SRC_TEST_PROCESSED = f'model_opennmt/run' + FOLDER  +f'/processed_test.{in_lang}.txt'
TGT_TEST_PROCESSED = f'model_opennmt/run' + FOLDER + f'/processed_test.{out_lang}.txt'

SRC_BPE_IN = f'model_opennmt/run' + FOLDER + f'/bpe_in.{in_lang}.txt'
SRC_BPE_OUT = f'model_opennmt/run' + FOLDER + f'/bpe_out.{in_lang}.txt'
SRC_BPE_CODES = f'model_opennmt/run' + FOLDER + f'/codes.{in_lang}.txt'
SRC_BPE_VOCAB = f'model_opennmt/run' + FOLDER + f'/bpe_vocab.{in_lang}.txt'

SRC_VLD_BPE_IN = f'model_opennmt/run' + FOLDER + f'/vld_bpe_in.{in_lang}.txt'
SRC_VLD_BPE_OUT = f'model_opennmt/run' + FOLDER + f'/vld_bpe_out.{in_lang}.txt'
SRC_VLD_BPE_CODES = f'model_opennmt/run' + FOLDER + f'/vld_codes.{in_lang}.txt'
SRC_VLD_BPE_VOCAB = f'model_opennmt/run' + FOLDER + f'/vld_bpe_vocab.{in_lang}.txt'

SRC_TEST_BPE_IN = f'model_opennmt/run' + FOLDER + f'/test_bpe_in.{in_lang}.txt'
SRC_TEST_BPE_OUT = f'model_opennmt/run' + FOLDER + f'/test_bpe_out.{in_lang}.txt'
SRC_TEST_BPE_CODES = f'model_opennmt/run' + FOLDER + f'/test_codes.{in_lang}.txt'
SRC_TEST_BPE_VOCAB = f'model_opennmt/run' + FOLDER + f'/test_bpe_vocab.{in_lang}.txt'

TGT_BPE_IN = TGT_PROCESSED
TGT_BPE_OUT = f'model_opennmt/run' + FOLDER + f'/bpe_out.{out_lang}.txt'
TGT_BPE_CODES = f'model_opennmt/run' + FOLDER + f'/codes.{out_lang}.txt'
TGT_BPE_VOCAB = f'model_opennmt/run' + FOLDER + f'/bpe_vocab.{out_lang}.txt'

TGT_VLD_BPE_IN = TGT_VLD_PROCESSED
TGT_VLD_BPE_OUT = f'model_opennmt/run' + FOLDER + f'/vld_bpe_out.{out_lang}.txt'
TGT_VLD_BPE_CODES= f'model_opennmt/run' + FOLDER + f'/vld_codes.{out_lang}.txt'
TGT_VLD_BPE_VOCAB = f'model_opennmt/run' + FOLDER + f'/vld_bpe_vocab.{out_lang}.txt'

TGT_TEST_BPE_IN = TGT_TEST_PROCESSED
TGT_TEST_BPE_OUT = f'model_opennmt/run' + FOLDER + f'/test_bpe_out.{out_lang}.txt'
TGT_TEST_BPE_CODES = f'model_opennmt/run' + FOLDER + f'/test_codes.{out_lang}.txt'
TGT_TEST_BPE_VOCAB = f'model_opennmt/run' + FOLDER + f'/test_bpe_vocab.{out_lang}.txt'

SRC_TEST = f'data/{domain}/test.{in_lang}.txt'
TGT_TEST = f'data/{domain}/test.{out_lang}.txt'

SRC_VLD = f'data/{domain}/validate.{in_lang}.txt'
TGT_VLD = f'data/{domain}/validate.{out_lang}.txt'

TGT_TEMP = f'model_opennmt/run' + FOLDER + f'/tgt_temp.{out_lang}.txt'
PREFIXES = f'model_opennmt/run' + FOLDER  + f'/prefixes.{in_lang}'
POSTFIXES = f'model_opennmt/run' + FOLDER  + f'/postfixes.{in_lang}'
WORDS = f'model_opennmt/run' + FOLDER  + f'/words.{in_lang}'
SUFFIXES = f'model_opennmt/run' + FOLDER  + f'/suffixes.{in_lang}'
ROOTS = f'model_opennmt/run' + FOLDER  + f'/roots.{in_lang}'
ENDINGS = f'model_opennmt/run' + FOLDER  + f'/endings.{in_lang}'

OUTPUT = f'model_opennmt/run' + FOLDER + f'/output.{out_lang}.txt'

DETOKEN_OUTPUT = f'model_opennmt/run' + FOLDER + f'/detoken_output.{out_lang}.txt'
DETOKEN_TGT = f'model_opennmt/run' + FOLDER + f'/detoken.{out_lang}.txt'

PIPELINE = 'model_opennmt/run' + FOLDER + '/pipeline.yaml'

PRPE_MULTI_TEMP = f'model_opennmt/run' + FOLDER + f'/prpe_mul_temp.{out_lang}.txt'

MODEL = f'model_opennmt/run' + FOLDER + f'/subword_model.{in_lang}.txt'

def is_windows():
    return os.name == 'nt'

# Deprecated
def clean_run():
    files = [f'model_opennmt/run/{in_lang}.vocab.src',
                f'model_opennmt/run/{out_lang}.vocab.src',
                f'model_opennmt/run/processed_train.{in_lang}.txt',
                f'model_opennmt/run/processed_train.{out_lang}.txt',
                f'model_opennmt/run/output.{in_lang}.txt']
    
    # Delete all models
    files = files + glob.glob('model_opennmt/run/*.pt')

    for file_path in files:
        try:
            os.remove(file_path)
        except OSError as e:
            print("Already deleted %s : %s" % (file_path, e.strerror))

def make_run_folder():
    os.mkdir('./model_opennmt/run' + FOLDER)

def copy_content(input_file, output_file):
    with open(input_file, encoding='utf8') as src_input, open(output_file, 'w', encoding='utf8') as src_out:
        src_arr = src_input.readlines()
        for src_line in src_arr:
            src_out.write(src_line)

        src_input.close()
        src_out.close()

def create_yaml(type):
    f = 'model_opennmt/run' + FOLDER
    with open(PIPELINE, 'w') as pl:
        pl.write('save_data: ' + f + '/result\n')

        pl.write('src_vocab: ' + f + f'/{in_lang}.vocab.src\n')
        pl.write('tgt_vocab: ' + f + f'/{out_lang}.vocab.src\n')

        pl.write('data:\n')
        pl.write('    corpus_1:\n')
        pl.write('        path_src: ' + SRC_PROCESSED + '\n')
        pl.write('        path_tgt: ' + TGT_PROCESSED + '\n')
        if opt.filter_too_long > -1:
            pl.write('        transforms: [filtertoolong]\n')
            pl.write(f'        src_seq_length: {opt.filter_too_long}\n')
            pl.write(f'        tgt_seq_length: {opt.filter_too_long}\n')
        pl.write('    valid:\n')
        pl.write('        path_src: ' + SRC_VLD_PROCESSED + '\n')
        pl.write('        path_tgt: ' + TGT_VLD_PROCESSED + '\n')

        pl.write('world_size: 1\n')
        pl.write('gpu_ranks: [0]\n')

        pl.write('batch-type: \"tokens\"\n')

        # Default is 10,000
        pl.write(f'valid_steps: {validate_steps}\n')
        pl.write(f'save_checkpoint_steps: {opt.save_steps}\n')
        pl.write(f'train_steps: {train_steps}\n') 
        pl.write(f'batch_size: {opt.batch_size}\n')
            
            # default type is tokens, default batch_size is 64

            # sgd is default optimizer

            # default dropout is 0.3

        if type == 'transformer':    

	        # Transformer specific batch settings
            #pl.write('queue_size: 10000\n')
            #pl.write('bucket_size: 32768\n')
            pl.write('valid_batch_size: 8\n')
            pl.write('accum_count: [4]\n')
            pl.write('accum_steps: [0]\n')

            # By default rnn is two layer, with 500 hidden units.
            # rnn default type is LSTM, uncomment this line for transformer
            pl.write('model_dtype: "fp32"\n')
	
            # default task is seq2seq (other option is lm)

            # sgd is the default optimizer. use adam for transformer
            pl.write('optim: "adam"\n')

            # Default dropout is 0.3. Use 0.1 for transformer
            pl.write('dropout: [0.1]\n')

            # default learning_rate is 1, 2 for transformer. learning_rate_decay is 0.5 and start_decay_steps is 50,000 
            # decay_steps is 10,000
            pl.write('learning_rate: 2\n')

	        # transformer specific optimization settings
            pl.write('warmup_steps: 8000\n')
            pl.write('decay_method: "noam"\n')
            pl.write('adam_beta2: 0.998\n')
            pl.write('max_grad_norm: 0\n')
            pl.write('label_smoothening: 0.1\n')
            pl.write('param_init: 0\n')
            pl.write('param_init_glorot: true\n')
            pl.write('normalization: "tokens"\n')

	        # transformer specific model settings
            pl.write('encoder_type: transformer\n')
            pl.write('decoder_type: transformer\n')
            pl.write('position_encoding: true\n')
            pl.write('enc_layers: 6\n')
            pl.write('dec_layers: 6\n')
            pl.write('heads: 8\n')
            pl.write('rnn_size: 256\n') #def 512 but i ran out of vram
            pl.write('word_vec_size: 256\n')
            pl.write('transformer_ff: 2048\n')
            pl.write('dropout_steps: [0]\n')
            pl.write('attention_dropout: [0.1]\n')
            
	
        pl.write('save_model: ' + f + '/model\n')

        pl.write('tensorboard: true\n')
        
        pl.close()

def tokenization_process():
    tokenization(TGT_INPUT, TGT_PROCESSED)
    tokenization(SRC_INPUT, SRC_PROCESSED)
    tokenization(TGT_VLD, TGT_VLD_PROCESSED)
    tokenization(SRC_VLD, SRC_VLD_PROCESSED)
    tokenization(SRC_TEST, SRC_TEST_PROCESSED)
    tokenization(TGT_TEST, TGT_TEST_PROCESSED)

def tokenization(source, output):
    tokenizer = MosesTokenizer(lang=f'{out_lang}')

    with open(source, encoding='utf8') as src_input, open(output, 'w', encoding='utf8') as src_out:
        src_arr = src_input.readlines()

        for src_line in src_arr:
            src_tok = tokenizer.tokenize(src_line)

            src_w = " ".join(src_tok)

            # Lower case the strings
            src_w = src_w.lower()

            src_out.write(src_w)

            src_out.write("\n")

        src_out.close()
        src_input.close()


def detokenization_process():
    detokenization(OUTPUT, DETOKEN_OUTPUT)
    detokenization(TGT_TEST_PROCESSED, DETOKEN_TGT)

def detokenization(source, out):
    detokenizer = MosesDetokenizer(f'{out_lang}')

    with open(source, encoding='utf8') as output, open(out, 'w', encoding='utf8') as detoken_out:
        output_arr = output.readlines()

        for line in output_arr:
            tokens = line.split(' ')

            detoken = detokenizer.detokenize(tokens)

            detoken_out.write(detoken)
            detoken_out.write("\n")

        output.close()
        detoken_out.close()

def prpe(src, out, train=True, apply=True):

    train_prpe = ['python', 'prpe/prpe6/learn_prpe.py', 
                  '-i',  src,
                  '-p',  PREFIXES,
                  '-r',  ROOTS, 
                  '-s',  SUFFIXES,
                  '-t',  POSTFIXES,
                  '-u',  ENDINGS,
                  '-w',  WORDS,
                  '-a', '32',
                  '-b', '500',
                  '-c', '500',
                  '-v', '500',
                  '-l', f'{in_lang}']

    apply_prpe = ['python', 'prpe/prpe6/apply_prpe.py', 
                  '-i',  src,
                  '-o',  out,
                  '-p',  PREFIXES,
                  '-r',  ROOTS, 
                  '-s',  SUFFIXES,
                  '-t',  POSTFIXES,
                  '-u',  ENDINGS,
                  '-w',  WORDS,
                  '-d', '0000',
                  '-m', '0',
                  '-n', '0',
                  '-l', f'{in_lang}']

    if train:
        subprocess.run(train_prpe)
    if apply:
        subprocess.run(apply_prpe)

def prpe_multi(src, out, train=True, apply=True):
    prpe(src, out, train, apply)

    iters = opt.prpe_multi_runs

    for i in range(iters):
        print(f"Iteration {i}")
        copy_content(out, PRPE_MULTI_TEMP)
        prpe(PRPE_MULTI_TEMP, out, train, apply)

def post_process_bpe(src, bpe_out):
    with open(bpe_out, encoding='utf8') as seg_out, open(src, 'w', encoding='utf8') as train_in:
        seg_arr = seg_out.readlines()

        for line in seg_arr:
            detoken = line.replace("@@", "")

            train_in.write(detoken)

        seg_out.close()
        train_in.close()

def bpe(src, bpe_in, bpe_codes, bpe_out):
    train_bpe = ['subword-nmt', 'learn-bpe', '--input', bpe_in, '--output', bpe_codes]
    test_bpe = ['subword-nmt', 'apply-bpe', '--codes', bpe_codes,'--input', bpe_in, '--output', bpe_out]

    subprocess.run(train_bpe)
    subprocess.run(test_bpe)

    post_process_bpe(src, bpe_out)

def process_unigram_list(list):
    string = ""
    for word in list:
        word = word.replace('▁', '')
        if word == ' ' or len(word.strip()) < 1:
            continue
        word = word.strip()
        string = string + ' ' + word
    return string[1:]

def unigram(src, out, train=True):

    if train:
        spm.SentencePieceTrainer.train(input=src, model_prefix=MODEL)
    with open(src, 'r', encoding='utf8') as in_file:
        src = in_file.readlines()
    sp = spm.SentencePieceProcessor(model_file=f'{MODEL}.model')
    lines = sp.encode(src, out_type=str)

    with open(out, 'w', encoding='utf8') as out_file:
        for line in lines:
            line = f'{process_unigram_list(line)}\n'
            out_file.write(line)
        out_file.close()

def segment_process(segment_type):
    
    if segment_type == 'prpe':
        # Segment source train
        prpe(SRC_PROCESSED, SRC_BPE_OUT)
        post_process_bpe(SRC_PROCESSED, SRC_BPE_OUT)

        # Segment source validation
        prpe(SRC_VLD_PROCESSED, SRC_VLD_BPE_OUT, train=False, apply=True)
        post_process_bpe(SRC_VLD_PROCESSED, SRC_VLD_BPE_OUT)

        # Segment source test
        prpe(SRC_TEST_PROCESSED, SRC_TEST_BPE_OUT, train=False, apply=True)
        post_process_bpe(SRC_TEST_PROCESSED, SRC_TEST_BPE_OUT)

    if segment_type == 'prpe_multi':
        # Segment source train
        prpe_multi(SRC_PROCESSED, SRC_BPE_OUT)
        post_process_bpe(SRC_PROCESSED, SRC_BPE_OUT)

        # Segment source validation
        prpe(SRC_VLD_PROCESSED, SRC_VLD_BPE_OUT, train=False, apply=True)
        post_process_bpe(SRC_VLD_PROCESSED, SRC_VLD_BPE_OUT)

        # Segment source test
        prpe(SRC_TEST_PROCESSED, SRC_TEST_BPE_OUT, train=False, apply=True)
        post_process_bpe(SRC_TEST_PROCESSED, SRC_TEST_BPE_OUT)

    if segment_type == 'prpe_bpe':

        # Segment source train
        prpe(SRC_PROCESSED, SRC_BPE_IN)
        bpe(SRC_PROCESSED, SRC_BPE_IN, SRC_BPE_CODES, SRC_BPE_OUT)

        # Segment source validation
        prpe(SRC_VLD_PROCESSED, SRC_VLD_BPE_IN, train=False, apply=True)
        bpe(SRC_VLD_PROCESSED, SRC_VLD_BPE_IN, SRC_VLD_BPE_CODES, SRC_VLD_BPE_OUT)

        # Segment source test
        prpe(SRC_TEST_PROCESSED, SRC_TEST_BPE_IN, train=False, apply=True)
        bpe(SRC_TEST_PROCESSED, SRC_TEST_BPE_IN, SRC_TEST_BPE_CODES, SRC_TEST_BPE_OUT)

    if segment_type == 'bpe':
        # Segment source train
        bpe(SRC_PROCESSED, SRC_PROCESSED, SRC_BPE_CODES, SRC_BPE_OUT)

        # Segment source validation
        bpe(SRC_VLD_PROCESSED, SRC_VLD_PROCESSED, SRC_VLD_BPE_CODES, SRC_VLD_BPE_OUT)

        # Segment source test
        bpe(SRC_TEST_PROCESSED, SRC_TEST_PROCESSED, SRC_TEST_BPE_CODES, SRC_TEST_BPE_OUT)

    if segment_type == 'unigram':
        unigram(SRC_PROCESSED, SRC_PROCESSED)
        unigram(SRC_VLD_PROCESSED, SRC_VLD_PROCESSED, train=False)
        unigram(SRC_TEST_PROCESSED, SRC_TEST_PROCESSED, train=False)

def metrics():
    
    with open(DETOKEN_OUTPUT, encoding='utf8') as output, open(DETOKEN_TGT, encoding='utf8') as reference:
        output_arr = output.readlines()
        ref_arr = [reference.readlines()]

        bleu = sacrebleu.corpus_bleu(output_arr, ref_arr).score
        chrf = sacrebleu.corpus_chrf(output_arr, ref_arr).score

        print(f'BLEU SCORE: {bleu}')
        print(f'CHRF SCORE: {chrf}')

        return bleu, chrf

def translate(model_name, source, i):
    translate = ['onmt_translate', '-model', 'model_opennmt/run' + FOLDER + model_name, '-src', SRC_TEST_PROCESSED, '-output', OUTPUT, '--replace_unk']
    print(f'Translating {opt.domain} {opt.model_type} + {opt.segment_type}: {in_lang}->{out_lang} at step {i}')
    subprocess.run(translate)
    detokenization_process()


def train():

    # Build vocabulary
    build = ['onmt_build_vocab', '-config', PIPELINE]
    subprocess.run(build)

    # Train model
    train = ['onmt_train', '-config', PIPELINE]
    subprocess.run(train)

def test():

    bleu_scores = dict()
    chrf_scores = dict()
    i = validate_steps

    # show performance on each validation step
    best = (0,0)
    while i <= train_steps:
        model_name = f'/model_step_{i}.pt'
        translate(model_name, SRC_VLD_PROCESSED, i)
        bleu_scores[i], chrf_scores[i] = metrics()

        if bleu_scores[i] > best[0]:
            best = bleu_scores[i], chrf_scores[i]
        i = i + validate_steps

    # eval on test set
    translate(f'/model_step_{train_steps}.pt', SRC_TEST_PROCESSED, train_steps)
    bleu_scores[i], chrf_scores[i] = metrics()
    

def pipeline():

    # Make run folder
    make_run_folder()

    # Generate yaml
    create_yaml(opt.model_type)

    # Tokenization pre-processing
    tokenization_process()

    # Segment source language
    segment_process(opt.segment_type)

    # Training
    train()

    # Testing
    test()

if __name__ == '__main__':
    pipeline()
