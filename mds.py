import numpy as np
import random
from sklearn.manifold import MDS
from config import *
from utils import get_experiment_name
from adaptive_experiment import AdaptiveExperiment
from adversarial_network import AdverserialNetwork
from final_resnet import FinalResnet
from nntools import StatsManager
from dataset import UTK
from dataset_design import DatasetDesign



def get_two_ref_target_imgs():
    target_dataset = UTK(TARGET_TRAIN_PATH, random_flips=True)
    image_indices = random.sample(list(range(len(target_dataset))), 2)
    return target_dataset[image_indices[0]], target_dataset[image_indices[1]]


def get_absolute_values(distance_matrix, reflabel1, reflabel2):
    mds = MDS(dissimilarity='precomputed', n_components=1)
    y = mds.fit_transform(distance_matrix)
    y = np.array([a[0] for a in y])
    print("mds output =",y)
    print("ref1 label = %s, ref2 label = %s"%(reflabel1.item(), reflabel2.item()))

    additive_scale = 0
    if ((y[-2] > y[-1]) ^ (reflabel1.item() > reflabel2.item())): #if both are true or both are false
        y = -1*y
    additive_scale =  reflabel1.item() -  y[-2]
    absolute_ages =  y + additive_scale
    return absolute_ages


def create_distance_matrix(dataset):
    experiment_name = get_experiment_name(EXPERIMENT_NAME, pairwise=True,
                                            ranking=True,
                                            adaptive=ADAPTIVE,
                                            loss = LOSS)

    dataset = list(dataset)
    net = FinalResnet()

    # if pretrained_model_name is not None:
    #     checkpoint_path = get_checkpoint_path(pretrained_model_name)
    #     data = torch.load(checkpoint_path, map_location=DEVICE)
    # else:
    #     data = None

    adver_net = AdverserialNetwork()
    stats_manager = StatsManager()


    exp = AdaptiveExperiment(net, adver_net, stats_manager,
                             output_dir=experiment_name,
                             perform_validation_during_training=False, pretrained_data=None)

    ref_img1, ref_img2 = get_two_ref_target_imgs()
    dataset.extend([ref_img1[0].unsqueeze(0)]+[ref_img2[0].unsqueeze(0)])

    num_samples = len(dataset)
    distance_matrix = np.zeros((num_samples, num_samples))
    
    exp.net.eval()
    exp.adv_net.eval()
    
    with torch.no_grad():
        for i in range(num_samples-1):
            for j in range(i+1,num_samples):
                x1_s, x2_s = dataset[i], dataset[j]
                if isinstance(x1_s, np.ndarray) and isinstance(x2_s, np.ndarray):
                    x1_s = torch.from_numpy(x1_s)
                    x2_s = torch.from_numpy(x2_s)
                x = torch.cat([x1_s, x2_s], dim=1)
                x = x.to(exp.net.device)
                f, y = exp.net.forward_adaptive(x)
                distance_matrix[i,j] = y[0][0]
                distance_matrix[j,i] = y[0][0]

    return distance_matrix, ref_img1[1], ref_img2[1]




if __name__ == '__main__':
    #_dataset = DatasetDesign(ETHNICITIES, SOURCE_VAL_SPLIT)
    x = UTK(TARGET_TRAIN_PATH, random_flips=True)
    dataset = [x[0], x[1], x[2], x[3]]
    dataset_nolabels = [y[0].unsqueeze(0) for y in dataset]
    print("labels of dataset",[y[1] for y in dataset])
    dm, ref_img1_lbl, ref_img2_lbl = create_distance_matrix(dataset_nolabels)
    mds_mat = get_absolute_values(dm,ref_img1_lbl, ref_img2_lbl )
    print(mds_mat)

