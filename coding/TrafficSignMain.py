
import torchvision.transforms as transforms

from torch.functional import F
from torch.utils.data import DataLoader


from TrafficSignDataset import TrafficSignDataset
from coding.modelAi import modelAi
from pytorchtools import EarlyStopping

class TrafficSignMain():

    __train_transform: transforms

    test_transform: transforms

    # DataLoader: is none, because some datasets don't have a test or valid set.
    train_dataloader: DataLoader = None
    valid_dataloader: DataLoader = None
    test_dataloader: DataLoader = None
    test_dataloader_unpoisoned: DataLoader = None

    def __init__(self, model: modelAi, epochs, image_size=32, batch_size=32) -> None:
        super().__init__()

        #self.visualizer = Visualizer()
        self.model = model
        #self.model.visualizer = self.visualizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.__create_train_transform(image_size)
        self.__create_test_transform(image_size)

        #self.hparams = HParams(name=model.name,)

    def start_tensorboard(self):
        from tensorboard import program
        from tensorboard.util import tb_logging
        tb = program.TensorBoard()
        tb.configure(argv=['', '--logdir', "../logs"])
        a = tb.launch()

        print("\n---------------------------------------------------------------------")
        print("\tSTARTING TENSORBOARD: {}".format(a))
        print("---------------------------------------------------------------------\n")

        tb_logging.get_logger().setLevel(level=40) #40 = Level.ERROR
        tb_logging.get_logger().disabled = True

    def __create_train_transform(self, image_size):
        self.__train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop((image_size, image_size), scale=(0.6, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.RandomAffine(15),
                transforms.RandomGrayscale(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], # pytorch nets values
                #                 std=[0.229, 0.224, 0.225]),
                transforms.ToTensor()
                # ,
                # transforms.Normalize(
                #   [0.3418, 0.3151, 0.3244],
                #  [0.1574, 0.1588, 0.1662] # Calculated with function, but the value changed for each dataset. So I fixed it with a static value.
                # )
            ]
        )

    def __create_test_transform(self, image_size):
        self.test_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ]
        )

    def creating_data(self, dataset=TrafficSignDataset, test_dir:str = None, train_dir:str = None, valid_dir:str = None, test_dir_unpoisoned:str = None ):
        if train_dir is not None:
            train_dataset = dataset(train_dir, self.__train_transform)
            self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            del train_dataset

        if valid_dir is not None:
            valid_dataset = dataset(valid_dir, self.test_transform)
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)
            del valid_dataset

        if test_dir is not None:
            test_dataset = dataset(test_dir, self.test_transform)
            self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
            del test_dataset

        if test_dir_unpoisoned is not None:
            test_dataset_unpoisoned = dataset(test_dir_unpoisoned, self.__train_transform)
            self.test_dataloader_unpoisoned = DataLoader(test_dataset_unpoisoned, batch_size=self.batch_size, shuffle=True)
            del test_dataset_unpoisoned
        print("Data creation complete.")

    def creating_data_for_ac(self, dataset=TrafficSignDataset,test_dir:str = None, train_dir:str = None, valid_dir:str = None, batch_size = 20):
        # Delete old dataset:

        if valid_dir is not None:
            #del self.valid_dataloader
            valid_dataset = dataset(valid_dir, self.test_transform)
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

        if test_dir is not None:
            #del self.test_dataloader
            test_dataset = dataset(test_dir, self.test_transform)
            self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        if train_dir is not None:
            #del self.train_dataloader
            train_dataset = dataset(train_dir, self.test_transform)
            self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Function for using the AI.

    def loading_ai(self, isPretrained: bool, should_train_if_needed=True, should_evaluate=True, verbose=True, patience=20):
        if self.model.did_save_model(self.model.name):
            if verbose:
                print("\n=> Found a saved model, start loading model.\n")
            last_epochs = self.model.load(verbose=verbose)
            if self.epochs != 0 and should_train_if_needed:
                if verbose:
                    print("Continue training at epoch: ", last_epochs)
                self.train_ai(self.epochs, last_epochs=last_epochs, verbose=verbose, patience=patience)
        else:
            if verbose:
                print(
                    "\n=> No model is saved. Creating File -> {}. Model will be saved during training.\n".format(
                        self.model. name))
            #self.model.run_training(current_epoch=0,epochs=self.epochs,train_loader=self.train_dataloader,val_loader=self.test_dataloader_unpoisoned) #mein code
            self.train_ai(self.epochs, verbose=verbose)

        if should_evaluate:
            self.evaluate_ai(verbose=verbose)

    def train_ai(self, epochs, last_epochs=0, patience=20, verbose=True):
        print(f"=>\tStart training AI on {self.model.device}")

        # initialize the early_stopping object
        early_stopping = EarlyStopping(model=self.model, patience=patience, verbose=True)

        for epoch in range(epochs):
            train_loss, train_acc = self.model.train(train_dataloader=self.train_dataloader, current_epoch=last_epochs + epoch)
            if verbose:
                print("=>\t[%d] loss: %.3f, accuracy: %.3f" % (epoch, train_loss, train_acc * 100) + "%")

            if self.valid_dataloader is not None:
                val_loss, val_pred_correct = self.model.evaluate_valid(dataloader=self.valid_dataloader, epoch=last_epochs + epoch)
                early_stopping(val_loss, model=self.model, epoch=epoch)

                if early_stopping.early_stop:
                    print('Early stopping')
                    break

                if verbose:
                    print("=>\tAccuracy on validation Dataset: %.3f" % (val_pred_correct * 100) + "% \n")
                self.model.scheduler.step(val_loss)
        print("=>\tFINISHED TRAINING")
    """
    def retraining_train_ai(self, epochs, isPretrained: bool, lastEpochs=0, verbose=True):
        print(f"=>\tStart training AI on {self.model.device}")

        for epoch in range(epochs):
            train_loss, train_acc = self.model.retrain(train_dataloader=self.train_dataloader,
                                                     isPretrained=isPretrained, current_epoch=lastEpochs + epoch)

            if verbose:
                print("=>\t[%d] loss: %.3f, accuracy: %.3f" % (epoch, train_loss, train_acc * 100) + "%")
            if not self.valid_dataloader == None:
                loss, pred_correct = self.model.evaluate_valid(dataloader=self.valid_dataloader, epoch=lastEpochs + epoch)#self.model.evaluate_valid(val_dataloader=self.valid_dataloader)
                if verbose:
                    print("=>\tAccuracy of validation Dataset: %.3f" % (pred_correct * 100) + "% \n")

        print("=>\tFINISHED TRAINING")
    """
    def evaluate_ai(self, verbose=True):
        # Werte Performance auf dem Testdatensatz aus, der entweder als clean oder poisonous angesehen wird, abhängig von der Eingabe in test_dir

        loss, pred_correct = self.model.evaluate_test(dataloader=self.test_dataloader)
        if verbose:
            print(f"loss on test dataset: {loss}")
            print(f"Accuracy of test Dataset: {pred_correct.__round__(3)} \n")

            if self.model.poisoned_data:
                print('BDSR classwise:')
                print(self.model.BDSR_classwise)

                # Evaluation of poisoned net on unpoisoned training data:
                loss, pred_correct = self.model.evaluate_test(dataloader=self.test_dataloader_unpoisoned)
                print('Performance of poisoned net on unpoisoned training data:')
                print(f"loss on test dataset: {loss}")
                print(f"Accuracy on test Dataset: {pred_correct.__round__(3)} \n")

    def get_activations(self, data_loader):

        return self.model.get_activations_of_last_hidden_layer(data_loader=TrafficSignMain.train_dataloader)

