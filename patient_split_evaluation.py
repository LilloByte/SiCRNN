import os, sys
import numpy as np
import torch
import librosa
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from networks import CRNN_2
from tqdm import tqdm
import gc
from sklearn.cluster import KMeans
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
torch.manual_seed(184)
torch.cuda.manual_seed(184)

ID = []
ACC = []
REC = []
PREC = []
SEP = []
dest_path = "/disks/disk1/dlillini/PSG_Audio/MELSP_6S/"
SR = 16000 # working sample rate
SELF = True  # True if I want to evaluate embeddings on self-train, False if I want to evaluate on untrained network

patiens = [ 995, 999, 1006,1000,1026,1088,1057, 1020, 1014, 1106, 1037,1008,1120, 1045, 1016, 1059, 1041, 
       1028, 1095, 1084, 1093, 1112, 1073, 1104, 1043, 1018,
        1082, 1022, 1110, 1039, 1089,  1116, 1010, 1097, 
       1108, 1069, 1118, 1086, 1024, 1071] # Patient number without leading zeros  

#SETTO ALCUNI IPERPARAMETRI DI INTERESSE
for P_n in patiens:
    print(f'Patient number:{P_n}')
    
    accuracy = 0.0
    prec = 0.0
    rec = 0.0
    
    checkpoint_path = "/home/dlillini/apnea_detection_v3/logs/exp_P" + str(P_n) + "/checkpoints"
    

    #Carico la rete CRNN_2 che deriva dalla CRNN con la differenza che restituisce l'embedding da [1,128] all'ultimo strato della GRU

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = CRNN_2()
    net = net.to(device)
    

    if SELF:
        net.load_state_dict(torch.load(os.path.join(checkpoint_path, "best_model.pth")))
        net.eval()


    #Carico i chunks melsp precedentemente creati e li invio alla rete.
    #Alla fine salvo tutti gli embeddings su due tensori pth per i successivi T-SNE e scatterplot.

    apnea_melsp = os.listdir(os.path.join(dest_path, 'P'+str(P_n), 'apnea'))
    nonapnea_melsp = os.listdir(os.path.join(dest_path, 'P'+str(P_n), 'nonapnea'))
    a_emb = torch.Tensor()
    na_emb = torch.Tensor()

    for a in tqdm(apnea_melsp):
        melsp = torch.load(os.path.join(dest_path, 'P'+str(P_n), 'apnea', a))
        melsp = melsp.to(device)
        with torch.no_grad():
            out = net(melsp.unsqueeze(0).unsqueeze(0))
        a_emb = torch.cat((a_emb,out.detach().cpu()))

    for na in tqdm(nonapnea_melsp):
        melsp = torch.load(os.path.join(dest_path, 'P'+str(P_n), 'nonapnea', na))
        melsp = melsp.to(device)
        with torch.no_grad():
            out = net(melsp.unsqueeze(0).unsqueeze(0))
        na_emb = torch.cat((na_emb,out.detach().cpu()))

    print(f'Embedded {a_emb.shape[0]} chunks of apnea')
    print(f'Embedded {na_emb.shape[0]} chunks of non-apnea')


    X_A = np.array(a_emb)
    X_NA = np.array(na_emb)
    X = np.vstack((X_A, X_NA))

    #K-mean 
    # load data
    
    
    lbl_apnea = 0
    lbl_nonapnea = 1
   

    y_A = lbl_apnea*np.ones(X_A.shape[0])
    y_NA = lbl_nonapnea*np.ones(X_NA.shape[0])
    y = np.vstack((y_A[:,np.newaxis], y_NA[:,np.newaxis]))
    y = np.squeeze(y)
    np.random.seed(5)
    estimator = KMeans(n_clusters=2)
    estimator.fit(X)
    labels_pred = estimator.labels_
    
    accuracy = ((labels_pred==y).sum() / len(y))
    rec = recall_score(y,labels_pred, average='binary')
    prec = precision_score(y,labels_pred, average='binary')

    cnt = estimator.cluster_centers_
    separability = np.linalg.norm(cnt[0]-cnt[1])

    if labels_pred[0] == 1:
        accuracy = 1.0-accuracy
        rec = 1.0-rec
        prec = 1.0-prec 

    # print(f'FIRST: {labels_pred[0]}')
    print(f'Accuracy: {accuracy}')
    print(f'Recall: {rec}')
    print(f'Precision: {prec}')
    
    ID.append(P_n)
    ACC.append(accuracy)
    REC.append(rec)
    PREC.append(prec)
    SEP.append(separability)

    torch.cuda.empty_cache()
    gc.collect()

dati = {
    'ID_patient': ID,
    'Accuracy': ACC,
    'Recall': REC,
    'Precision': PREC,
    'Separability': SEP
}
my_file = '/home/dlillini/apnea_detection_v3/self_train_result.csv'
df = pd.DataFrame(dati)
df.to_csv(my_file, index=False)

