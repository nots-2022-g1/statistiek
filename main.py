import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

files = [
    'L4H256',
    'L4H512'
]

NamedHistory = namedtuple('NamedHistory', 'name data')

history = [NamedHistory(filename, np.load(filename + '.npy', allow_pickle=True).item()) for filename in files]
# print(np.load('L4H256.npy', allow_pickle='TRUE'))
# PLOT ACCURACY AND LOSS OVER TIME
# history = [
#     np.load('L4H256.npy', allow_pickle='TRUE').item(),
#     np.load('L4H512.npy', allow_pickle='TRUE').item()
# ]
# history_dict = np.load('L4H256.npy', allow_pickle='TRUE').item()
# history_dict2 = np.load('L4H512.npy', allow_pickle='TRUE').item()
# print(history_dict.keys())
#
# acc = history_dict['binary_accuracy']
# acc2 = history_dict2['binary_accuracy']
# val_acc = history_dict['val_binary_accuracy']
# val_acc2 = history_dict2['val_binary_accuracy']
# loss = history_dict['loss']
# loss2 = history_dict2['loss']
# val_loss = history_dict['val_loss']
# val_loss2 = history_dict2['val_loss']

epochs = range(1, len(history[0].data['binary_accuracy']) + 1)
fig = plt.figure(figsize=(20, 12))
fig.tight_layout()

# plt.subplot(2, 1, 1)
# plt.plot(epochs, loss, label='L4H256 training loss')
# plt.plot(epochs, loss2, label='L4H512 training loss')
# plt.plot(epochs, val_loss, label='L4H256 validation loss')
# plt.plot(epochs, val_loss2, label='L4H512 validation loss')
# plt.title('Training and validation loss')
# # plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.subplot(2, 1, 2)
for dataset in history:
    plt.plot(epochs, dataset.data['binary_accuracy'], label=dataset.name + ' acc')
    plt.plot(epochs, dataset.data['val_binary_accuracy'], label=dataset.name + ' val acc')
    # plt.plot(epochs, dataset.data['loss'], label=dataset.name + ' loss')
    # plt.plot(epochs, dataset.data['val_loss'], label=dataset.name + ' val loss')

# plt.plot(epochs, acc, label='L4H256 training acc')
# plt.plot(epochs, acc2, label='L4H512 training acc')
# plt.plot(epochs, val_acc, label='L4H256 validation acc')
# plt.plot(epochs, val_acc2, label='L4H512 validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower left')
plt.grid()

plt.show()
