import torch
from torch import nn
from torch import optim
from dataset import UTK
from dataset_design import DatasetDesign
from torch.utils.data import DataLoader
from archive import model_selection

src = '/Users/vulcan/da-age-prediction/Data/utk/'
ethnicity = {
    "source": 0,
    "target": 1,
}

source_val_split_perc = 80
_dataset = DatasetDesign(src, ethnicity, source_val_split_perc)
train_dataset = UTK('/Users/vulcan/da-age-prediction/Data/source_train.csv')  #Todo: replace with destination value, remove hardcoding
val_dataset = UTK('/Users/vulcan/da-age-prediction/Data/source_validation.csv')
num_workers = 0
model = model_selection.model
criterion = nn.MSELoss()

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=model_selection.batch_size,
                          num_workers=num_workers,
                          shuffle=True)

val_laoder = DataLoader(dataset=val_dataset,
                        batch_size=model_selection.batch_size,
                        num_workers=num_workers,
                        shuffle=True)

model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=model_selection.learning_rate)
use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()

curr_epoch = 0
epoch_losses = []
val_losses = []
total_epochs = 2
def train():
    global curr_epoch
    global total_epochs
    while(curr_epoch < total_epochs):
        model.train()
        _curr_epoch_loss = []
        try:
            for iter, (x_img, label_age) in enumerate(train_loader):

                optimizer.zero_grad()

                if use_gpu:
                    input = x_img.cuda()
                    label = label_age.cuda()
                else:
                    input, label = x_img, label_age
                output = model(input)
                loss = criterion(output, label)
                _curr_epoch_loss.append(loss.item())
                loss.backward()
                optimizer.step()
        except Exception as e:
            print('Exception ', str(e))



        curr_epoch_loss = sum(_curr_epoch_loss) / len(_curr_epoch_loss)
        epoch_losses.append(curr_epoch_loss)
        print('train loss after epoch: {}, is {}', curr_epoch, curr_epoch_loss)

    val_loss = val(curr_epoch)
    val_losses.append(val_loss)
    curr_epoch = curr_epoch + 1

def val():
    model.eval()
    val_losses = []
    for iter, (x_img_val, label_age_val) in enumerate(val_laoder):
        if use_gpu:
            input_val = x_img_val.cuda()
            label_val = label_age_val.cuda()
        else:
            input_val, label_val = label_age_val
        output_val = model(input_val)
        loss_val = criterion(output_val, label_val)
        val_losses.append(loss_val.item())
    total_loss = sum(val_losses)/len(val_losses)
    print('validation loss after epoch: {}, is {}', curr_epoch, sum(val_losses) / len(val_losses))
    return total_loss

if __name__ == "__main__":
    # try:
    #     model.cuda()
    #     print('starting with epoch : ', curr_epoch)
    # except:
    #     print('exception')
    #     pass
    train()
    # plt.figure()
    # x_axis = [i for i in range(0, total_epochs)]
    # plt.plot(x_axis, epoch_losses)
    # plt.plot(x_axis, val_losses)
    # plt.gca().legend(('train','validation'))
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Loss vs Epochs')
    # plt.savefig('./loss.png')



