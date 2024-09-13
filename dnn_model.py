
import datetime
import os
import random
import string
import traceback

import keras.layers
import keras.models
import keras.utils
import numpy
import pandas


char_to_int = {c: i+1 for c, i in zip(string.ascii_letters, range(26))}
char_to_int['_'] = 27
int_to_char = {v: k for k, v in char_to_int.items()}


class DNNModel(object):
    """Deep Neural Networks Model to learn the Hangman game."""

    def __init__(self, train_dict_src: str, model_weights_src: str, verbose: bool):
        self.train_dict = None           # a list of words from words_250000_train.txt
        self.max_word_size = 29          # max length of words in the dictionary
        self.model = None                # the DNN model
        self.train_hist_encoded = None   # a list of tuples: (masked_word, guessed_letters, probs of the answer letters)
        self.train_hist_debug = None
        self.train_hist_label = train_dict_src.replace('.txt', '').split('_')[-1]

        # build dictionary for training
        if not os.path.exists(train_dict_src):
            raise ValueError(f'ERROR: training dictionary source file {train_dict_src} does not exist')

        with open(train_dict_src, 'r') as file:
            self.train_dict = file.read().splitlines()

        # self.max_word_size = max([len(x) for x in self.train_dict])

        # CNN+LSTM state machine for masked words as 1x(max_word_size) arrays filled by letter codes:
        # 1-26 for a-z, 27 for _, 0 for padding, a total of 28 dimensions. E.g. 'x_z' -> [0, ..., 0, 24, 27, 26]
        masked_st = keras.models.Sequential(layers=[
            keras.layers.Input(shape=(self.max_word_size,)),
            keras.layers.Embedding(input_dim=28, output_dim=32, mask_zero=True, input_length=self.max_word_size),  # min model: output_dim=4
            keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),  # min model: filters=4
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.LSTM(units=100, dropout=0.2, recurrent_dropout=0.2),  # min model: units=6
            keras.layers.Dense(units=60, activation='relu'),                   # min model: units=6
        ], name='MaskedWordModel')

        # MLP state machine for guessed letters as a 1x26 array filled by 0's and 1's.
        # E.g. ['a', 'c'] -> [1, 0, 1, 0, ..., 0]
        guessed_st = keras.models.Sequential(layers=[
            keras.layers.Input(shape=(26,)),
            keras.layers.Dense(units=40, activation='relu'),  # min model: units=4
            keras.layers.Dense(units=40, activation='relu'),  # min model: units=4
        ], name='GuessedLettersModel')

        # final DNN Model
        x = keras.layers.Concatenate()([masked_st.output, guessed_st.output])
        x = keras.layers.Dense(units=100, activation='relu')(x)  # min model: units=4
        x = keras.layers.Dense(units=26, activation='softmax')(x)
        self.model = keras.models.Model(inputs=[masked_st.input, guessed_st.input], outputs=x, name='DNNModel')

        # load weights if the file exists in order to predict without training
        if model_weights_src is not None and os.path.exists('model_weights.h5'):
            self.model.load_weights(f"""{model_weights_src}""")
            print(f"""\n--- loaded model weights from {model_weights_src}\n""")
        else:
            print('\n--- model weights not loaded\n')

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        if verbose:
            self.model.summary()
            print(f"""\n--- training dictionary has {len(self.train_dict)} words with max word size of {max(list(map(len, self.train_dict)))}\n""")
            keras.utils.plot_model(self.model, to_file='model_specs.png', show_shapes=True, show_dtype=True,
                                   show_layer_names=True, expand_nested=True, show_layer_activations=True)
            model_config = self.model.to_json()
            with open("model_config.json", "w") as json_file:
                json_file.write(model_config)

    def _save_train_hist(self, masked_word: list[str], guessed_letters: list[str], word: str) -> None:
        """Collect the history of state machines during one game to generate training data."""

        self.train_hist_debug.append(f"""{word},{''.join(masked_word)},{''.join(guessed_letters)}""")

        masked_encoded = keras.utils.pad_sequences([[char_to_int[x] for x in masked_word]],
                                                   maxlen=self.max_word_size, dtype='float32')

        guessed_encoded = numpy.zeros(shape=26)
        for x in guessed_letters:
            guessed_encoded[char_to_int[x] - 1] = 1.0
        guessed_encoded = guessed_encoded.reshape((1, 26))

        # the filtering problem: given the state machines of guessed letters and masked word, estimate the probability
        # distribution of the full word in order to minimize the categorical cross entropy
        probs = numpy.zeros(shape=26)
        for x in word:
            probs[char_to_int[x] - 1] = 1.0
        probs = numpy.array([0 if guessed_encoded[0][i] > 0 else x for i, x in enumerate(probs)])
        probs = probs / probs.sum()

        self.train_hist_encoded.append((masked_encoded, guessed_encoded, probs))

    def predict(self, masked_word: str, guessed_letters: list[str]) -> str:
        masked_input = keras.utils.pad_sequences([[char_to_int[x] for x in masked_word]],
                                                 maxlen=self.max_word_size, dtype='float32')

        guessed_input = numpy.zeros(shape=26)
        for x in guessed_letters:
            guessed_input[char_to_int[x]-1] = 1.0
        guessed_input = guessed_input.reshape((1, 26))

        probs = self.model.predict([masked_input, guessed_input], verbose=0).flatten()
        for i in probs.argsort()[::-1]:
            c = int_to_char[i+1]
            if c not in guessed_letters:
                return c

    def train(self, n_epochs: int,  batch_size: int) -> None:
        self.train_hist_encoded = []
        self.train_hist_debug = []

        for i in range(n_epochs):
            print(f"""--- epoch: {i}/{n_epochs}\n""")

            words_trained = 0
            for word in self.train_dict:
                tries_remains = 6
                masked_word = ['_'] * len(word)
                guessed_letters = []

                # save the initial states for training to estimate the first character
                self._save_train_hist(masked_word=masked_word, guessed_letters=guessed_letters, word=word)

                while (tries_remains > 0) and ('_' in masked_word):
                    guess_letter = self.predict(masked_word=''.join(masked_word), guessed_letters=guessed_letters)
                    guessed_letters.append(guess_letter)

                    if guess_letter in word:
                        for k in range(len(word)):
                            if word[k] == guess_letter:
                                masked_word[k] = guess_letter
                    else:
                        tries_remains -= 1

                    # save changes of the state machines (and the ensuing probability estimation) till a game's end
                    if '_' in masked_word:
                        self._save_train_hist(masked_word=masked_word, guessed_letters=guessed_letters, word=word)

                words_trained += 1

                if (words_trained % batch_size == 0) or (words_trained == len(self.train_dict)):
                    with open(f'train_data_debug_{self.train_hist_label}.csv', 'a') as debug:
                        debug.write('\n'.join(self.train_hist_debug) + '\n')
                    self.train_hist_debug = []

                    masked_encoded, guessed_encoded, probs = zip(*self.train_hist_encoded)
                    self.train_hist_encoded = []

                    masked_encoded = numpy.vstack(list(masked_encoded)).astype(float)
                    guessed_encoded = numpy.vstack(list(guessed_encoded)).astype(float)
                    probs = numpy.vstack(probs)

                    history = self.model.train_on_batch([masked_encoded, guessed_encoded], probs)

                    x = pandas.DataFrame(numpy.hstack((masked_encoded, guessed_encoded, probs)))
                    x.to_csv(f'train_data_encoded_{self.train_hist_label}.csv', mode='a', index=False, header=False)

                # saving model weights as check points
                if (words_trained % 1000 == 0) or (words_trained == len(self.train_dict)):
                    print(f"""\t{datetime.datetime.now()}, trained words: {words_trained}/{len(self.train_dict)}""")
                    self.model.save_weights(f"""model_weights_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.h5""")

        self.model.save_weights(f"""model_weights_final_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.h5""")
        self.model.save_weights(f"""model_weights.h5""")


class HangmanAPI(object):

    def __init__(self, train_dict_src: str, model_weights_src: str):
        self.guessed_letters = []
        self.train_dict = None
        self.dnn_model = DNNModel(train_dict_src=train_dict_src, model_weights_src=model_weights_src, verbose=False)

        with open(train_dict_src, 'r') as file:
            self.train_dict = file.read().splitlines()

    def guess(self, masked_word: str) -> str:
        c = self.dnn_model.predict(masked_word=masked_word, guessed_letters=self.guessed_letters)
        return c

    def start_game(self, verbose: bool) -> str:
        self.guessed_letters = []

        tries_remains = 6
        word = numpy.random.choice(self.train_dict)
        masked_word = ['_'] * len(word)

        print(f"""--- Game Starts: {tries_remains} tries remain, word is '{word}', masked_word is '{"".join(masked_word)}'""")

        while (tries_remains > 0) and ('_' in masked_word):
            guess_letter = self.guess(masked_word=''.join(masked_word))
            self.guessed_letters.append(guess_letter)

            if guess_letter in word:
                for i in range(len(word)):
                    if word[i] == guess_letter:
                        masked_word[i] = guess_letter
            else:
                tries_remains -= 1

            if verbose:
                print(f"""\tguessed {guess_letter}, masked_word is now {''.join(masked_word)}, {tries_remains} tries remain(s), guessed letters are {self.guessed_letters}""")

        if tries_remains == 0:
            print(f"""--- You Lose: masked_word is now '{''.join(masked_word)}', guessed letters are {self.guessed_letters}\n""")
            return 'LOSE'
        else:
            print(f"""You win! masked_word is now '{''.join(masked_word)}'\n""")
            return 'WIN'


if __name__ == '__main__':
    try:
        # train the model by using full dictionary and generating inputs & outputs on the fly; single_run is to
        # facilitate bash script run.sh to drive the training repeatedly, one data file at a time
        if False:
            single_run = True

            curr_dir = os.path.dirname(os.path.realpath(__file__))

            with open('log_dnn_model.txt', 'r') as f:
                used_files = f.read().splitlines()
                used_files = [x for x in used_files if 'finished' in x]

            # the directory train_dict_shuffled_samples contains shuffled samples of the full dictionary, stored in
            # multiple files
            data_dir = os.path.join(curr_dir, 'train_dict_shuffled_samples')
            all_files = [os.path.abspath(os.path.join(data_dir, f)) for f in os.listdir(data_dir)]
            unused_files = list()
            for f in all_files:
                if sum([(f in used) for used in used_files]) == 0:
                    unused_files.append(f)

            if len(unused_files) > 0 and single_run:
                unused_files = unused_files[0:1]

            for f in unused_files:
                print(f'\n{f}')
                with open('log_dnn_model.txt', 'a') as log:
                    log.write(f"""{datetime.datetime.now()}: started training on {f}\n""")

                # each round of training will update and reload model_weights.h5
                dnn_model = DNNModel(train_dict_src=f"""{f}""", model_weights_src='model_weights.h5', verbose=True)
                dnn_model.train(n_epochs=1, batch_size=50)

                with open('log_dnn_model.txt', 'a') as log:
                    log.write(f"""{datetime.datetime.now()}: finished training on {f}\n""")

                hangman = HangmanAPI(train_dict_src='words_250000_train.txt', model_weights_src='model_weights.h5')
                n_games = 100
                wins = 0

                for i in range(n_games):
                    print(f"""{i}/{n_games}""")
                    if 'WIN' == hangman.start_game(verbose=False):
                        wins += 1

                accuracy = wins / n_games * 100
                print(f"""{datetime.datetime.now()}: {round(accuracy,2)}% is the test accuracy on {f}\n""")
                with open('log_dnn_model.txt', 'a') as log:
                    log.write(f"""{datetime.datetime.now()}: {round(accuracy,2)}% is the test accuracy on {f}\n""")

        if True:
            n_tests = 1
            results = list()

            hangman = HangmanAPI(train_dict_src='words_250000_train.txt', model_weights_src='model_weights.h5')
            for n in range(n_tests):

                print(f"""{n+1}/{n_tests}""")
                if 'WIN' == hangman.start_game(verbose=True):
                    results.append(1)
                else:
                    results.append(0)

            results = numpy.array(results)
            m = results.mean()
            std = numpy.sqrt(m*(1-m)/n_tests)
            print(f"""95% confidence: {round(m * 100, 2)}% +/- 2x{round(std * 100, 2)}%""")

        # test DNNModel.predict(...)
        if False:
            dnn_model = DNNModel(train_dict_src='words_250000_train.txt',
                                 model_weights_src='model_weights.h5',
                                 verbose=True)
            print(dnn_model.predict('__sk', ['b', 'x']))

        if False:
            print(char_to_int)
            print(int_to_char)

        # randomly choose n_words words from training dictionary and write to words.txt
        if False:
            with open('words_250000_train.txt', 'r') as f:
                full_dict = f.read().splitlines()

            random.shuffle(full_dict)
            print(full_dict[:10], '\n', full_dict[-10:])

            n_words = 5000
            with open('words.txt', 'w') as f:
                f.write('\n'.join(full_dict[:n_words]))

        # split the training dictionary into several files
        if False:
            with open('words_250000_train.txt', 'r') as f:
                full_dict = f.read().splitlines()

            random.shuffle(full_dict)

            n_words = 2000
            n_files = len(full_dict) // n_words + 1
            for i in range(n_files):
                start = i * n_words
                end = (i + 1) * n_words
                samples = full_dict[start:end]
                if i < 10:
                    name = f"""00{i}"""
                elif i < 100:
                    name = f"""0{i}"""
                else:
                    name = f"""{i}"""
                print(f'generating words_{name}.txt: rows {start}-{end}')
                with open(f'words_{name}.txt', 'w') as f:
                    f.write('\n'.join(samples))

    except Exception as err:
        print(traceback.format_exc())
