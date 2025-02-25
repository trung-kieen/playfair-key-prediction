
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input , LSTM


############################## Return only last hidden state ##################################
n_timestep = 4 # Number of element in sequence
n_feature = 4 # Number of property
LSTMCellAmount = 1
input = Input(shape = (n_timestep, n_feature))

output = LSTM(LSTMCellAmount)(input)
model = Model(inputs=input, outputs=[output])
# print(model.summary())

"""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)             │ (None, 4, 4)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm (LSTM)                          │ (None, 1)                   │              24 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘

Coclution:
input(shape=(4, 4)) then output shape will be (None, 4, 4)
only final hidden state in this example will return return in lstm layer
"""
# Result


################################## Return all hidden state #####################################


n_timestep = 4 # Number of element in sequence
n_feature = 4 # Number of property
LSTMCellAmount = 1
input = Input(shape = (n_timestep, n_feature))

output = LSTM(LSTMCellAmount, return_sequences = True)(input)
model = Model(inputs=input, outputs=[output])
# print(model.summary())

"""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_layer_1 (InputLayer)           │ (None, 4, 4)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_1 (LSTM)                        │ (None, 4, 1)                │              24 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
"""




############################## Return hidden state twice and all cell state ######################
# n_timestep = 4 # Number of element in sequence
# n_feature = 4 # Number of property
# LSTMCellAmount = 1
# input = Input(shape = (n_timestep, n_feature))
# lstm, state_h, state_c= LSTM(LSTMCellAmount, return_state = True)(input)
# model = Model(inputs=input, outputs=[lstm, state_h, state_c])
# print(model.summary())



numberOfLSTMcells= 1
n_timestep = 4 # Number of element in sequence
n_feature = 10 # Number of property
input =Input(shape=(n_timestep, n_feature))
lstm, state_h, state_c = LSTM(numberOfLSTMcells, return_state = True)(input)
model = Model(inputs=input, outputs=[lstm, state_h, state_c])
print(model.summary())
