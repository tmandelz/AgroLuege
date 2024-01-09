import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import numpy as np
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn
import argparse
from src.dlbs.datasetdlbs import Dataset_DLBS
from models.multi_stage_sequenceencoder import multistageSTARSequentialEncoder
from models.networkConvRef import model_2DConv
from eval import evaluate_fieldwise


def main(
        datadir=None,
        fold_num=None,
        batchsize=1,
        snapshot=None,
        layer=6,
        hidden=64,
        stage=3,
        gt_path=None,
        input_dim=None,
        num_workers=0,
):

    testdataset = Dataset_DLBS(datadir, 0., 'test',True, 4, gt_path, num_channel=4, apply_cloud_masking=False,small_train_set_mode=False)
    nclasses = testdataset.n_classes
    nclasses_local_1 = testdataset.n_classes_local_1
    nclasses_local_2 = testdataset.n_classes_local_2
    print('Num classes:', nclasses)

    # Class stage mappping
    s1_2_s3 = testdataset.l1_2_g
    s2_2_s3 = testdataset.l2_2_g

    # Define the model
    network = multistageSTARSequentialEncoder(24, 24, nstage=stage, nclasses=nclasses,
                                              nclasses_l1=nclasses_local_1, nclasses_l2=nclasses_local_2,
                                              input_dim=input_dim, hidden_dim=hidden, n_layers=layer, cell='star',
                                              wo_softmax=True)
    network_gt = model_2DConv(nclasses=nclasses, num_classes_l1=nclasses_local_1, num_classes_l2=nclasses_local_2,
                          s1_2_s3=s1_2_s3, s2_2_s3=s2_2_s3,
                          wo_softmax=True, dropout=1)
    network.eval()
    network_gt.eval()

    model_parameters = filter(lambda p: p.requires_grad, network.parameters())
    model_parameters2 = filter(lambda p: p.requires_grad, network_gt.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters]) + sum([np.prod(p.size()) for p in model_parameters2])
    print('Num params: ', params)

    if torch.cuda.is_available():
        network = torch.nn.DataParallel(network).cuda()
        network_gt = torch.nn.DataParallel(network_gt).cuda()

    if snapshot is not None:
        checkpoint = torch.load(snapshot)
        network.load_state_dict(checkpoint['network_state_dict'])
        network_gt.load_state_dict(checkpoint['network_gt_state_dict'])

    evaluate_fieldwise(network, network_gt, testdataset, batchsize=batchsize, level=1, fold_num=fold_num,workers=num_workers)
    evaluate_fieldwise(network, network_gt, testdataset, batchsize=batchsize, level=2, fold_num=fold_num,workers=num_workers)
    evaluate_fieldwise(network, network_gt, testdataset, batchsize=batchsize, level=3, fold_num=fold_num,workers=num_workers)