import numpy as np
import matplotlib.pyplot as plt

log_path = 'logs'
losses = np.load(log_path + '/loss_hist.npy')
accs1 = np.load(log_path + '/acc1_hist.npy')
accs2 = np.load(log_path + '/acc2_hist.npy')
val_losses = np.load(log_path + '/val_loss.npy')
val_accs1 = np.load(log_path + '/val_acc1.npy')
val_accs2 = np.load(log_path + '/val_acc2.npy')

plt.plot(np.arange(len(losses)), losses, label = "train_loss")
plt.plot(np.arange(len(val_losses)), val_losses, label = "validation_loss")
plt.legend()
plt.xlabel('num_epochs')
plt.ylabel('value')
plt.savefig(log_path + "/loss_graph")
plt.show()
plt.close()

# plt.plot(np.arange(len(val_losses)), val_losses, label = "validation_loss")
# plt.legend()
# plt.xlabel('num_epochs')
# plt.ylabel('value')
# plt.savefig(log_path + "/val_loss_graph")
# plt.show()
# plt.close()

plt.plot(np.arange(len(accs1)), accs1, label = "bacteria train accuracy")
plt.plot(np.arange(len(val_accs1)), val_accs1, label = "bacteria validation accuracy")
# plt.plot(np.arange(len(val_accs1)), val_accs1, label = "val_bact_acc")
# plt.plot(np.arange(len(val_accs2)), val_accs2, label = "val_swab_acc")
plt.xlabel('num_epochs')
plt.ylabel('value')
plt.legend()
plt.savefig(log_path + "/bact_acc_graph")
plt.show()
plt.close()


plt.plot(np.arange(len(accs2)), accs2, label = "swab train accuracy")
plt.plot(np.arange(len(val_accs2)), val_accs2, label = "swab validation accuracy")
plt.xlabel('num_epochs')
plt.ylabel('value')
plt.legend()
plt.savefig(log_path + "/swab_acc_graph")
plt.show()
plt.close()
