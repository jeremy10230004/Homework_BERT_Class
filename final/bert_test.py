from transformers import BertTokenizer
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import TFBertForSequenceClassification
import numpy as np
from tensorflow.python.keras.utils import np_utils


tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
with tpu_strategy.scope():
    data_path = "../input/bert-test/SingleLabel.csv"

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    df = pd.read_csv(data_path)
    print(df.head(5))
    print("---csv read---")
    lyrics = df["lyrics"]
    label = df["label"]

    emo_dict = {"Sadness": 0, "Tension": 1, "Tenderness": 2}
    final_label = []
    for i in df["label"]:
        final_label.append(emo_dict[i])

    #y = np_utils.to_categorical(final_label , 3)  # 3種類!!!
    y = np.array(final_label)
    #print(y)
    print(y.shape)
    
    s_rate = int(y.shape[0] * 0.90)
    
    train_x, test_x = lyrics[:s_rate], lyrics[s_rate:]
    train_y, test_y = y[:s_rate], y[s_rate:]
    print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
    

    # --------------------------------------------------------
    def convert_example_to_feature(review):
        return tokenizer.encode_plus(review,
                                     add_special_tokens=True,  # add [CLS], [SEP]
                                     max_length=30,  # max length of the text that can go to BERT
                                     pad_to_max_length=True,  # add [PAD] tokens
                                     return_attention_mask=True,  # add attention mask to not focus on pad tokens
                                     )


    # map to the expected input to TFBertForSequenceClassification, see here
    def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
        return {
                   "input_ids": input_ids,
                   "token_type_ids": token_type_ids,
                   "attention_mask": attention_masks,
               }, label


    def encode_examples(x, y, limit=-1):
        # prepare list, so that we can build up final TensorFlow dataset from slices.
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        label_list = []
        if (limit > 0):
            ds = ds.take(limit)

            
        for lyrics in x:
            bert_input = convert_example_to_feature(lyrics)

            input_ids_list.append(bert_input['input_ids'])
            token_type_ids_list.append(bert_input['token_type_ids'])
            attention_mask_list.append(bert_input['attention_mask'])

        return tf.data.Dataset.from_tensor_slices(
            (input_ids_list, attention_mask_list, token_type_ids_list, y)).map(map_example_to_dict)
    # ------------------------------------------------------------

    learning_rate = 3e-5
    batch_size = 64
    number_of_epochs = 8
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,epsilon=1e-08, clipnorm=1)
    
    # train dataset
    ds_train_encoded = encode_examples(train_x, train_y).shuffle(10000).batch(batch_size)
    # test dataset
    ds_test_encoded = encode_examples(test_x, test_y).batch(batch_size)

    # ----------------------------------------------------------

    model = TFBertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=3)
    print("model_build")
    # -----------------------------------------------------------



    # model initialization
    model = TFBertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=3)

    # optimizer Adam recommended
    

    # we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    # fit model
    bert_history = model.fit(ds_train_encoded, epochs=number_of_epochs, batch_size = batch_size)
    # evaluate test set
    model.evaluate(ds_test_encoded)
