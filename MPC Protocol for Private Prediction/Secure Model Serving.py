from collections import OrderedDict
import numpy as np
import tensorflow as tf
import tf_encrypted as tfe

print(tf.__version__)

num_classes = 6
input_shape = (1, 64, 64, 3)

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(64, (3, 3), input_shape=(64, 64, 3), padding='same', activation='relu'),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
  tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
  tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
  tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',),
  tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
  tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',),
  tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',),
  tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',),
  tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
  tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',),
  tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',),
  tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',),
  tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(6, name='logit')
])

pre_trained_weights = 'model_VGG13_avg_16_w1.h5'
model.load_weights(pre_trained_weights)

players = OrderedDict([
    ('server0', 'localhost:4000'),
    ('server1', 'localhost:4001'),
    ('server2', 'localhost:4002'),
])

for player_name in players.keys():
    print("python -m tf_encrypted.player --config /Users/shahbazsiddeeq/PhD/PPML/tfe.config {}".format(player_name))

config = tfe.RemoteConfig(players)
config.save('./tfe.config')

config.connect_servers()
tfe.set_config(config)
tfe.set_protocol(tfe.protocol.SecureNN())


with tfe.protocol.SecureNN():
    tfe_model = tfe.keras.models.clone_model(model)

# Set up a new tfe.serving.QueueServer for the shared TFE model
q_input_shape = (1, 64, 64, 3)
q_output_shape = (1, 6)

server = tfe.serving.QueueServer(
    input_shape=q_input_shape, output_shape=q_output_shape, computation_fn=tfe_model
)

request_ix = 1

def step_fn():
    global request_ix
    print("Served encrypted prediction {i} to client.".format(i=request_ix))
    request_ix += 1

server.run(num_steps=3, step_fn=step_fn)
# You are ready to move to the Private Prediction Client notebook to request some private predictions.


process_ids = !ps aux | grep '[p]ython -m tf_encrypted.player --config' | awk '{print $2}'
for process_id in process_ids:
    !kill {process_id}
    print("Process ID {id} has been killed.".format(id=process_id))
