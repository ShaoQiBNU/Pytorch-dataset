################# load packages #################
import torch
import torch.nn
import torch.optim
import torch.utils.data
import model
from tqdm import tqdm
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
import GQResnet
import GQResNeXt
import GQSEResNet
import GQSEResNeXt
import GQSKnet
import GQDensenet

###################### train and test #######################
class train_and_test():

    def __init__(self, num_epochs, learning_rate, class_size, data_loader_train, data_loader_test, log_interval, version_name ):

        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.class_size = class_size
        self.data_loader_test = data_loader_test
        self.data_loader_train = data_loader_train
        self.log_interval = log_interval
        self.version_name = version_name

    ################# train and test #################
    def train_epoch(self):

        ########### model ###########
        Model = model.GQNet()
        #Model = GQResnet.resnet18()
        #Model = GQResNeXt.resnext50_32x4d()
        #Model = GQSEResNet.se_resnet_18()
        #Model = GQSEResNeXt.se_resnext_50()
        #Model = GQSKnet.SKNet50()
        #Model = GQDensenet.densenet121()

        ########### optimizer, loss ###########
        optimizer = torch.optim.Adam(Model.parameters(), lr=self.learning_rate)
        loss = torch.nn.CrossEntropyLoss()

        ########### create trainer and evaluator ###########
        trainer = create_supervised_trainer(
            Model, optimizer, loss
        )

        evaluator = create_supervised_evaluator(
            Model, metrics={
                "accuracy": Accuracy(),
                "loss": Loss(loss)
            }
        )

        ########### log ###########
        desc = "ITERATION - loss: {:.2f}"
        pbar = tqdm(
            initial=0, leave=False, total=len(self.data_loader_train),
            desc=desc.format(0)
        )

        ########### train loss ###########
        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            iter = (engine.state.iteration - 1) % len(self.data_loader_train) + 1

            if iter % self.log_interval == 0:
                pbar.desc = desc.format(engine.state.output)
                pbar.update(self.log_interval)

        ########### train results ###########
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            pbar.refresh()
            evaluator.run(self.data_loader_train)
            metrics = evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_cross_loss = metrics['loss']

            tqdm.write(
                "Training Results - Epoch: {}  Avg accuracy: {:.2f}  Avg loss: {:.2f}"
                    .format(engine.state.epoch, avg_accuracy, avg_cross_loss)
            )

        ########### test results ###########
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(self.data_loader_test)
            metrics = evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_cross_loss = metrics['loss']

            tqdm.write(
                "Validation Results - Epoch: {}  Avg accuracy: {:.2f}  Avg loss: {:.2f}"
                    .format(engine.state.epoch, avg_accuracy, avg_cross_loss))

            pbar.n = pbar.last_print_n = 0

        ########### trainer start ###########
        trainer.run(self.data_loader_train, max_epochs=self.num_epochs)
        #torch.save(model, str(self.version_name))
        pbar.close()