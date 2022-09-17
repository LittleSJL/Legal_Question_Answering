import os
import sys
import random
import argparse
import collections
import tensorflow as tf
path_cwd = os.getcwd()
path_tools = os.path.join(path_cwd, 'pipeline')
sys.path.append(path_tools)
from reader import bert_config, modeling, tokenization
from reader.bert_nlp import model_fn_builder, FeatureWriter, input_fn_builder, write_predictions
from reader.input_handler import read_data_reader, convert_examples_to_features, read_test_pipeline

parser = argparse.ArgumentParser()

# train the model/use the model to predict
parser.add_argument('--train', action='store_true')
parser.add_argument('--predict', action='store_true')

# Test the pipeline - given the paragraphs retrieved by retriever
# or test only reader - given the correct paragraphs
parser.add_argument('--test_pipeline', help='Test the pipeline or only reader', default=True)

# training config
parser.add_argument('--human', default=True)  # if adding human questions into training set
parser.add_argument('--meaningless', default=False)  # if adding meaningless QG questions into training set
parser.add_argument('--train_refor', default=True)  # if adding reformulation questions into training set
parser.add_argument('--human_only', default=False)  # if only add human questions in the training set

# testing config
parser.add_argument('--test_refor', default=True)  # if adding reformulation questions into test set

# Here, I will only use 2_128 small BERT model for an example.
# 12_768 BERT-base model can be downloaded from https://github.com/google-research/bert
parser.add_argument('--model_dir', type=str, default='model/reader/2_128/model.ckpt')
parser.add_argument('--vocab_file', type=str, default='model/reader/vocab.txt')
parser.add_argument('--bert_config_file', type=str, default='model/reader/2_128/bert_config.json')

# output_dir: output folder for results
parser.add_argument('--output_dir', type=str, default='output/reader')
args = parser.parse_args()


pipeline_selection = args.test_pipeline
training_config = {'human_selection': args.human, 'meaningless_selection': args.meaningless,
                   'refor_selection': args.train_refor, 'human_only': args.human_only}
test_config = {'refor_selection': args.test_refor}

FLAGS = bert_config.FinetuningConfig(model_dir=args.model_dir,
                                     vocab_file=args.vocab_file,
                                     bert_config_file=args.bert_config_file,
                                     training_config=training_config, test_config=test_config,
                                     output_dir=args.output_dir,
                                     do_train=args.train, do_predict=args.predict)

tf.logging.set_verbosity(tf.logging.INFO)
bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
tokenizer = tokenization.FullTokenizer(
    vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

tpu_cluster_resolver = None
is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    master=FLAGS.master,
    model_dir=FLAGS.output_dir,
    save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=FLAGS.iterations_per_loop,
        num_shards=FLAGS.num_tpu_cores,
        per_host_input_for_training=is_per_host))

train_examples = None
num_train_steps = None
num_warmup_steps = None

if FLAGS.do_train:
    train_examples = read_data_reader(is_training=True, human_selection=FLAGS.training_config['human_selection'],
                                      meaningless_selection=FLAGS.training_config['meaningless_selection'],
                                      refor_selection=FLAGS.training_config['refor_selection'],
                                      human_only=FLAGS.training_config['human_only'])
    print('Nums of training examples:', len(train_examples))
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # Pre-shuffle the input to avoid having to make a very large shuffle buffer in in the `input_fn`.
    rng = random.Random(12345)
    rng.shuffle(train_examples)

model_fn = model_fn_builder(
    bert_config=bert_config,
    init_checkpoint=FLAGS.init_checkpoint,
    learning_rate=FLAGS.learning_rate,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=FLAGS.use_tpu,
    use_one_hot_embeddings=FLAGS.use_tpu)

# If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=FLAGS.use_tpu,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=FLAGS.train_batch_size,
    predict_batch_size=FLAGS.predict_batch_size)

if FLAGS.do_train:
    # We write to a temporary file to avoid storing very large constant tensors in memory.

    train_writer = FeatureWriter(
        filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
        is_training=True)
    convert_examples_to_features(
        examples=train_examples,
        tokenizer=tokenizer,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        is_training=True,
        output_fn=train_writer.process_feature)
    train_writer.close()

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num orig examples = %d", len(train_examples))
    tf.logging.info("  Num split examples = %d", train_writer.num_features)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    del train_examples

    train_input_fn = input_fn_builder(
        input_file=train_writer.filename,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)

    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])

if FLAGS.do_predict:
    if pipeline_selection:
        # if testing pipeline, use paragraphs retrieved by retriever
        eval_examples = read_test_pipeline(refor_selection=FLAGS.test_config['refor_selection'])
    else:
        # else, use the correct paragraph
        eval_examples = read_data_reader(is_training=False, refor_selection=FLAGS.test_config['refor_selection'])
    eval_writer = FeatureWriter(
        filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
        is_training=False)
    eval_features = []


    def append_feature(feature):
        eval_features.append(feature)
        eval_writer.process_feature(feature)


    convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        is_training=False,
        output_fn=append_feature)
    eval_writer.close()

    tf.logging.info("***** Running predictions *****")
    tf.logging.info("  Num orig examples = %d", len(eval_examples))
    tf.logging.info("  Num split examples = %d", len(eval_features))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_input_fn = input_fn_builder(
        input_file=eval_writer.filename,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

    # If running eval on the TPU, you will need to specify the number of steps.
    all_results = []
    for result in estimator.predict(
            predict_input_fn, yield_single_examples=True):
        if len(all_results) % 1000 == 0:
            tf.logging.info("Processing example: %d" % (len(all_results)))
        unique_id = int(result["unique_ids"])
        start_logits = [float(x) for x in result["start_logits"].flat]
        end_logits = [float(x) for x in result["end_logits"].flat]
        all_results.append(
            RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits))

    print('Number of eval examples:', len(eval_examples))
    if pipeline_selection:
        file_name = "prediction_pipeline.json"
    else:
        file_name = "prediction_upper.json"
    output_prediction_file = os.path.join(FLAGS.output_dir, "predictions.json")
    output_nbest_file = os.path.join(FLAGS.output_dir, file_name)
    output_null_log_odds_file = os.path.join(FLAGS.output_dir, "null_odds.json")

    write_predictions(eval_examples, eval_features, all_results,
                      FLAGS.n_best_size, FLAGS.max_answer_length,
                      FLAGS.do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file,
                      FLAGS)
