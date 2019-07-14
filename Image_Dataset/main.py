###################### load packages ####################
import Dataset
import torch
import config
import train

###################### main函数 ####################
def main():
    ########### 读取配置文件 ##########
    ch = config.ConfigHandler("./config.ini")
    ch.load_config()

    ########### 读取参数 ##########
    train_batch_size = int(ch.config["model"]["train_batch_size"])
    test_batch_size = int(ch.config["model"]["test_batch_size"])

    num_epochs = int(ch.config["model"]["num_epochs"])
    learning_rate = float(ch.config["model"]["learning_rate"])
    class_size = int(ch.config["model"]["class_size"])

    ########### 读取log和model ##########
    log_interval = int(ch.config["log"]["log_interval"])
    version_name = ch.config["log"]["version_name"]

    train_file = ch.config["data"]["train_file"]
    test_file = ch.config["data"]["test_file"]

    ########### 获取训练数据loader ##########
    data_train = Dataset.ImageDataset(train_file, train=True)
    data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=train_batch_size, shuffle=True)

    ########### 获取测试数据loader ##########
    data_test = Dataset.ImageDataset(test_file, train=False)
    data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=test_batch_size, shuffle=False)

    ########### 训练和评价 ##########
    train.train_and_test(num_epochs, learning_rate, class_size, data_loader_train, data_loader_test,
                         log_interval, version_name).train_epoch()


if __name__ == "__main__":
    main()
