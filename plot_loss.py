import os
from matplotlib import pyplot as plt



def get_train_dev_losses(src_dir, filename):
    log_path = os.path.join(src_dir, filename)
    # read log file lines
    lines = open(log_path).readlines()

    train_losses = []
    dev_losses = []
    for line in lines:
        if "Epoch Average Loss" in line:
            # get traing loss
            a, b = line.split("Epoch Average Loss")
            loss = b.split("-")[0].strip()
            train_losses.append(float(loss))
        elif "Eval Devset" in line:
            # Sample from log file:   Epoch #13: Average Loss 0.66763 - Epoch Acc: 76.49000 - Epoch Testing Time: 0.035 min(s)
            pieces = line.split("-")
            dev_loss = float(pieces[0].split("Loss")[-1].strip())
            dev_losses.append(dev_loss)
    
    return train_losses, dev_losses


if __name__ == "__main__":
    filename = "CNN-LSTM-greedy.txt"
    src_dir = "training_logs"
    train_losses, dev_losses = get_train_dev_losses(src_dir, filename)

    filename2 = "CNN-GRU-greedy.txt"
    src_dir = "training_logs"
    train_losses2, dev_losses2 = get_train_dev_losses(src_dir, filename2)
    
    filename3 = "VIT-LSTM-GREEDY.txt"
    src_dir = "training_logs"
    train_losses3, dev_losses3 = get_train_dev_losses(src_dir, filename3)

    filename4 = "VIT-GRU-GREEDY.txt"
    src_dir = "training_logs"
    train_losses4, dev_losses4 = get_train_dev_losses(src_dir, filename4)


    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize =(19,11))
    model_1_name = filename.split(".")[0]
    model_2_name = filename2.split(".")[0]
    model_3_name = filename3.split(".")[0]
    model_4_name = filename4.split(".")[0]

    plt.plot(range(len(train_losses)), train_losses, color="blue", label=model_1_name+" train_loss")
    plt.plot(range(len(train_losses2)), train_losses2, color="red", label=model_2_name+" train_loss")
    plt.plot(range(len(train_losses3)), train_losses3, color="purple", label=model_3_name+" train_loss")
    plt.plot(range(len(train_losses4)), train_losses4, color="pink", label=model_4_name+" train_loss")

    plt.plot(range(len(dev_losses)), dev_losses, color="green", label=model_1_name+" dev_loss")
    plt.plot(range(len(dev_losses2)), dev_losses2, color="orange", label=model_2_name+" dev_loss")
    plt.plot(range(len(dev_losses3)), dev_losses3, color="black", label=model_3_name+" dev_loss")
    plt.plot(range(len(dev_losses4)), dev_losses4, color="brown", label=model_4_name+" dev_loss")

    plt.legend(loc="lower left", prop={'size': 14})
    plt.title("ResNet|ViT + LSTM|GRU Train and Dev Loss")
    plt.xlabel = "epoch"
    plt.ylabel = "loss"
    plt.grid()
    plt.savefig("train_dev_loss_plot.jpg", bbox_inches='tight')