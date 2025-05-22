import os, sys
from tqdm import tqdm
import torch
import json
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from networks import Siamese_CRNN
from networks import Siamese_CRNN_4
from networks import Siamese_CRNN_2
from losses import ContrastiveLoss
from dataset import ApneaDataset_all



def start_train():
   
    #Selezione degli iperparametri
   
   
    print(f"kernel_size: {ker_size}, n_conv: {n_conv}, hidden_gru: {gru}, input: {inp}")
    
    parameters = {
    "kernel_size": ker_size,
    "n_conv": n_conv,
    "hidden_gru": gru,
    "input": inp
}
    with open(percorso_file, 'w') as file:
        json.dump(parameters, file)
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(log_dir=os.path.join(waights_path, str(exp_n)))
    model = None  
    # define model########################
    if n_conv == 2:
        if inp == 8:
            model = Siamese_CRNN_2(kernel_size=ker_size, pool_size1=(4,2), pool_size2=(2,2), gru=gru)
        if inp == 12:
            model = Siamese_CRNN_2(kernel_size=ker_size, pool_size1=(4,2), pool_size2=(3,2), gru=gru)
        if inp == 18:
            model = Siamese_CRNN_2(kernel_size=ker_size, pool_size1=(6,2), pool_size2=(2,2), gru=gru)
        if inp == 25:
            model = Siamese_CRNN_2(kernel_size=ker_size, pool_size1=(5,2), pool_size2=(5,2), gru=gru)
        if inp == 32:
            model = Siamese_CRNN_2(kernel_size=ker_size, pool_size1=(8,2), pool_size2=(4,2), gru=gru)
    if n_conv == 3:
        if inp == 8:
            model = Siamese_CRNN(kernel_size=ker_size, pool_size1=(2,2), pool_size2=(2,2), pool_size3=(2,2), gru=gru)
        if inp == 12:
            model = Siamese_CRNN(kernel_size=ker_size, pool_size1=(3,2), pool_size2=(2,2), pool_size3=(2,2), gru=gru)
        if inp == 18:
            model = Siamese_CRNN(kernel_size=ker_size, pool_size1=(4,2), pool_size2=(2,2), pool_size3=(2,2), gru=gru)
        if inp == 25:
            model = Siamese_CRNN(kernel_size=ker_size, pool_size1=(5,2), pool_size2=(2,2), pool_size3=(2,2), gru=gru)
        if inp == 32:
            model = Siamese_CRNN(kernel_size=ker_size, pool_size1=(8,2), pool_size2=(2,2), pool_size3=(2,2), gru=gru)
    if n_conv == 4:
        if inp == 8:
            model = Siamese_CRNN_4(kernel_size=ker_size, pool_size1=(2,2), pool_size2=(2,2), pool_size3=(2,2), pool_size4=(1,2), gru=gru)
        if inp == 12:
            model = Siamese_CRNN_4(kernel_size=ker_size, pool_size1=(3,2), pool_size2=(2,2), pool_size3=(2,2), pool_size4=(1,2), gru=gru)
        if inp == 18:
            model = Siamese_CRNN_4(kernel_size=ker_size, pool_size1=(2,2), pool_size2=(2,2), pool_size3=(2,2), pool_size4=(2,2), gru=gru)
        if inp == 25:
            model = Siamese_CRNN_4(kernel_size=ker_size, pool_size1=(2,2), pool_size2=(2,2), pool_size3=(2,2), pool_size4=(3,2), gru=gru)
        if inp == 32:
            model = Siamese_CRNN_4(kernel_size=ker_size, pool_size1=(4,2), pool_size2=(2,2), pool_size3=(2,2), pool_size4=(2,2), gru=gru)
        
    ######################################################
    model = model.to(device)
    torch.manual_seed(7) #fisso il seed per l'inizializzazione dei pesi (seed=7 per riprodurre gli esperimenti )
    torch.cuda.manual_seed(7)
    # loss and optimizer
    criterion = ContrastiveLoss(margin = 2.0).to(device)
    # optimizer = optim.Adam(model.parameters(), lr = 5e-4, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 1e-2)
    optimizer = optim.SGD(model.parameters(), lr = 5e-3)

    # defining train dataset and dataloader    
    train_set = ApneaDataset_all(data_path)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    # Defining validation set and dataloader
    
    print(f'Start experiment')
    print(f'Found device: {device}\n')

    start_tstamp = datetime.now()

   

    
    best_loss = 1e6

    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}/{num_epochs}')
        model.train()
        training_loss = 0.0
        # Training loop
        print(f'train...')
        for  batch in tqdm(train_loader):
            optimizer.zero_grad()
            S_1, S_2, lbl_1, lbl_2 = batch
            S_1 = S_1[:,:,0:inp,:]
            S_2 =  S_2[:,:,0:inp,:]
            E_1, E_2 = model(S_1.to(device), S_2.to(device))
            # dissim is the "dissimilarity" label: 1 (True) = different inputs, 0 (False) = same input class
            # it is computed with XOR       
            dissim = lbl_1 ^ lbl_2
            loss = criterion(E_1, E_2, dissim.to(device))
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

            
        if training_loss < best_loss:
            torch.save(model.state_dict(), os.path.join(waights_path, str(exp_n), 'best_model.pth'))
            best_loss = training_loss
            print(f'Loss: {best_loss}')
        writer.add_scalar("train_loss", training_loss,epoch+1)     
        writer.flush()
        
        
    

    end_tstamp = datetime.now()
    writer.close()
    print(f'Training started at {start_tstamp.strftime("%m/%d/%Y - %H:%M:%S")}, Training finished at {end_tstamp.strftime("%m/%d/%Y - %H:%M:%S")}')


if __name__ == "__main__":
    #######################
    exp_n = '1'
    ker_size = (5,5)
    n_conv = 4
    gru = 32
    inp = 12
    #######################
    
    num_epochs = 100 
    waights_path = "/home/dlillini/esercitazioni_2025/DACLS_apnea/apnea_detection/checkpoints"
    data_path = "/home/dlillini/DATA/PSG_Audio/new_split/split1/train/"
    
    
    
    percorso_file = os.path.join(waights_path, str(exp_n),'parameters.json')

    if not os.path.exists(os.path.join(waights_path, str(exp_n))):
            os.makedirs(os.path.join(waights_path, str(exp_n)))
        
        
    
    start_train()
        
 
    