import json
import pickle

import numpy as np 
from tensorflow import keras


from etc.model.config import * # hprams


class Brain:

	def __init__(self):
		super().__init__()

		self.model = keras.Model.load_model(model_path)
		self.inferred = []

		with open('etc/model/saved/tokenizer.pickle', 'rb') as handle:
			self.tokenizer = pickle.load(handle)

		with open('etc/model/saved/lbl_encoder.pickle', 'rb') as handle:
			self.lbl_encoder = pickle.load(handle)

		with open('etc/model/saved/data.pickle', 'rb') as handle:
			self.data = pickle.load(handle)

	
	def chat(self, text):
		self.text = text.lower()

		self.result = self.model.predict(keras.preprocessing.sequence.pad_sequences(self.tokenizer.texts_to_sequences([inp]),
	                                             truncating='post', maxlen=max_len))
		self.tag = self.lbl_encoder.inverse_transform([np.argmax(result)])

		for i in data['intents']:
			if i['tag'] == self.tag:
				self.inferred.append(np.random.choice(i['responses']))
				self.inferred.append(i['context_set'])
		
		return [self.text, self.inferred]

	
	