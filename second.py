import numpy as np
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences

# Sample sprint data and text summaries
sprint_data = np.array([[20, 18, 22, 19, 21, 20], [1, 2, 1, 3, 1, 2], [2, 1, 2, 3, 1, 2]])
text_summaries = ["Velocity increased by 2 story points compared to the previous sprint.",
                  "Scope volatility was low this sprint.",
                  "Commitment reliability improved to high this sprint."]

# Define vocabulary
vocab = {'<PAD>': 0, '<START>': 1, '<END>': 2, 'Velocity': 3, 'Scope': 4, 'Volatility': 5,
         'was': 6, 'low': 7, 'this': 8, 'sprint': 9, 'Commitment': 10, 'Reliability': 11,
         'improved': 12, 'to': 13, 'high': 14, 'compared': 15, 'by': 16, 'story': 17, 'points': 18, 'previous': 19}

# Define Seq2Seq model architecture
class Seq2SeqModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super(Seq2SeqModel, self).__init__()
        self.encoder = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
        self.decoder = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    def call(self, inputs, training=False):
        encoder_inputs, decoder_inputs = inputs
        encoder_embedded = self.embedding(encoder_inputs)
        decoder_embedded = self.embedding(decoder_inputs)
        encoder_outputs, state_h, state_c = self.encoder(encoder_embedded, training=training)
        decoder_outputs, _, _ = self.decoder(decoder_embedded, initial_state=[state_h, state_c], training=training)
        return decoder_outputs

# Hyperparameters
vocab_size = len(vocab)
embedding_dim = 256
units = 1024
max_length = 10
batch_size = 64
epochs = 10

# Prepare data
encoder_input_data = pad_sequences(sprint_data, padding='post')
decoder_input_data = np.array([[vocab['<START>']]] * len(text_summaries))
decoder_target_data = pad_sequences([[vocab[word] for word in summary.split()] for summary in text_summaries], padding='post')

# Build and train the model
model = Seq2SeqModel(vocab_size, embedding_dim, units)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs)

# Generate text summaries for new sprint data
new_sprint_data = np.array([[18, 19, 20, 21, 22, 23], [3, 1, 2, 1, 3, 1], [1, 2, 3, 2, 1, 2]])
encoder_input_data_new = pad_sequences(new_sprint_data, padding='post')
decoder_input_data_new = np.array([[vocab['<START>']]] * len(new_sprint_data))

output_tokens = model.predict([encoder_input_data_new, decoder_input_data_new])
for i in range(len(output_tokens)):
    output_text = [list(vocab.keys())[list(vocab.values()).index(token)] for token in np.argmax(output_tokens[i], axis=1) if token != 0]
    print(' '.join(output_text))
