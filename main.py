import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

files = [
    'L4H256',
    'L4H512',
    'albert',
    'L12H768'
]

NamedHistory = namedtuple('NamedHistory', 'name data')

history = [NamedHistory(filename, np.load(filename + '.npy', allow_pickle=True).item()) for filename in files]

epochs = range(20)
fig = plt.figure(figsize=(20, 12))
fig.tight_layout()

for dataset in history:
    plt.plot(epochs, dataset.data['binary_accuracy'], label=dataset.name + ' acc')
    plt.plot(epochs, dataset.data['val_binary_accuracy'], label=dataset.name + ' val acc')
    # plt.plot(epochs, dataset.data['loss'], label=dataset.name + ' loss')
    # plt.plot(epochs, dataset.data['val_loss'], label=dataset.name + ' val loss')

plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid()

plt.show()
