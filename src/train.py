import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn
import argparse
import os
from src.dataset import Dataset
from models.multi_stage_sequenceencoder import multistageSTARSequentialEncoder
from models.networkConvRef import model_2DConv
from src.eval import evaluate_fieldwise
import wandb
import numpy as np

def main(
        datadir=None,
        batchsize=4,
        workers=12,
        epochs=1,
        lr=1e-3,
        snapshot=None,
        checkpoint_dir=None,
        weight_decay=0.0000,
        name='debug',
        layer=6,
        hidden=64,
        lrS=1,
        lambda_1=1,
        lambda_2=1,
        stage=3,
        clip=1,
        seed=None,
        fold_num=None,
        gt_path=None,
        cell=None,
        dropout=None,
        input_dim=None,
        apply_cm=None,
        project="test",
        run_group="test",
        model_architectur = "ms-convstar"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    np.random.seed(seed)
    torch.manual_seed(seed)

    traindataset = Dataset(datadir, 0., 'train', False, fold_num, gt_path, num_channel=input_dim, apply_cloud_masking=apply_cm)
    testdataset = Dataset(datadir, 0., 'test', True, fold_num, gt_path, num_channel=input_dim, apply_cloud_masking=apply_cm)

    nclasses = traindataset.n_classes
    nclasses_local_1 = traindataset.n_classes_local_1
    nclasses_local_2 = traindataset.n_classes_local_2

    # set weights close to zero
    LOSS_WEIGHT = torch.ones(nclasses)
    LOSS_WEIGHT[0] = 10**-40 
    LOSS_WEIGHT_LOCAL_1 = torch.ones(nclasses_local_1)
    LOSS_WEIGHT_LOCAL_1[0] = 10**-40
    LOSS_WEIGHT_LOCAL_2 = torch.ones(nclasses_local_2)
    LOSS_WEIGHT_LOCAL_2[0] = 10**-40

    # Class stage mappping
    s1_2_s3 = traindataset.l1_2_g
    s2_2_s3 = traindataset.l2_2_g

    traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=batchsize, shuffle=True, num_workers=workers)

    
    
    network = multistageSTARSequentialEncoder(24, 24, nstage=stage, nclasses=nclasses,
                                                  nclasses_l1=nclasses_local_1, nclasses_l2=nclasses_local_2,
                                                  input_dim=input_dim, hidden_dim=hidden, n_layers=layer, cell=cell,
                                                  wo_softmax=True)
    network_gt = model_2DConv(nclasses=nclasses, num_classes_l1=nclasses_local_1, num_classes_l2=nclasses_local_2,
                              s1_2_s3=s1_2_s3, s2_2_s3=s2_2_s3,
                              wo_softmax=True, dropout=dropout)

    model_parameters = filter(lambda p: p.requires_grad, network.parameters())
    model_parameters2 = filter(lambda p: p.requires_grad, network_gt.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters]) + sum([np.prod(p.size()) for p in model_parameters2])
    print('Num params: ', params)

    optimizer = torch.optim.Adam(list(network.parameters()) + list(network_gt.parameters()), lr=lr,
                                 weight_decay=weight_decay)

    loss = torch.nn.CrossEntropyLoss(weight=LOSS_WEIGHT)
    loss_local_1 = torch.nn.CrossEntropyLoss(weight=LOSS_WEIGHT_LOCAL_1)
    loss_local_2 = torch.nn.CrossEntropyLoss(weight=LOSS_WEIGHT_LOCAL_2)

    if lrS == 1:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=-1)
    elif lrS == 2:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, last_epoch=-1)
    elif lrS == 3:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5, last_epoch=-1)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, last_epoch=-1)


    print('CUDA available: ', torch.cuda.is_available())
    if torch.cuda.is_available():
        network = torch.nn.DataParallel(network).cuda()
        network_gt = torch.nn.DataParallel(network_gt).cuda()
        loss = loss.cuda()
        loss_local_1 = loss_local_1.cuda()
        loss_local_2 = loss_local_2.cuda()

    start_epoch = 0
    best_test_acc = 0

    if snapshot is not None:
        checkpoint = torch.load(snapshot)
        network.load_state_dict(checkpoint['network_state_dict'])
        network_gt.load_state_dict(checkpoint['network_gt_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizerA_state_dict'])

    WBrun = setup_wandb_run(project,run_group,fold_num,lr,epochs,model_architectur,workers,batchsize,seed)

    for epoch in range(start_epoch, epochs):
        print("\nEpoch {}".format(epoch+1))

        train_epoch(traindataloader, network, network_gt, optimizer, loss, loss_local_1, loss_local_2,
                    lambda_1=lambda_1, lambda_2=lambda_2, stage=stage, grad_clip=clip,epoch=epoch)

        # call LR scheduler
        lr_scheduler.step()

        if epoch % 1 == 0:
            test_acc = evaluate_fieldwise(network, network_gt, testdataset,epoch=epoch, batchsize=batchsize,workers=0,n_epochs=epochs)
            if checkpoint_dir is not None:
                checkpoint_name = os.path.join(checkpoint_dir, name + '_epoch_' + str(epoch) + "_model.pth")
                if test_acc > best_test_acc:
                    print('Model saved! Best val acc:', test_acc)
                    best_test_acc = test_acc
                    torch.save({'network_state_dict': network.state_dict(),
                                'network_gt_state_dict': network_gt.state_dict(),
                                'optimizerA_state_dict': optimizer.state_dict()}, checkpoint_name)
    evaluate_fieldwise(network, network_gt, testdataset, batchsize=batchsize,epoch=epochs ,level=1, fold_num=fold_num,workers=workers,n_epochs=epochs)
    evaluate_fieldwise(network, network_gt, testdataset, batchsize=batchsize,epoch=epochs, level=2, fold_num=fold_num,workers=workers,n_epochs=epochs)


def train_epoch(dataloader, network, network_gt, optimizer, loss, loss_local_1, loss_local_2, lambda_1,
                lambda_2, stage, grad_clip,epoch):

    network.train()
    network_gt.train()

    mean_loss_glob = 0.
    mean_loss_local_1 = 0.
    mean_loss_local_2 = 0.
    mean_loss_gt = 0.
    for iteration, data in enumerate(dataloader):
        n +=1
        optimizer.zero_grad()

        input, target_glob, target_local_1, target_local_2 = data

        if torch.cuda.is_available():
            input = input.cuda()
            target_glob = target_glob.cuda()
            target_local_1 = target_local_1.cuda()
            target_local_2 = target_local_2.cuda()

        output_glob, output_local_1, output_local_2 = network.forward(input)
        l_glob = loss(output_glob, target_glob)
        l_local_1 = loss_local_1(output_local_1, target_local_1)
        l_local_2 = loss_local_2(output_local_2, target_local_2)

        if stage == 3 or stage == 1:
            total_loss = l_glob + lambda_1 * l_local_1 + lambda_2 * l_local_2
        elif stage == 2:
            total_loss = l_glob + lambda_2 * l_local_2
        else:
            total_loss = l_glob

        mean_loss_glob += l_glob.data.cpu().numpy()
        mean_loss_local_1 += l_local_1.data.cpu().numpy()
        mean_loss_local_2 += l_local_2.data.cpu().numpy()
        wandb.log({"iteration": iteration,
                  "epoch": epoch, "loss global": l_glob, "loss local 1": l_local_1,"loss local 2": l_local_2,"total loss":total_loss})

        # Refinement -------------------------------------------------
        output_glob_R = network_gt([output_local_1, output_local_2, output_glob])
        l_gt = loss(output_glob_R, target_glob)
        mean_loss_gt += l_gt.data.cpu().numpy()

        total_loss = total_loss + l_gt

        wandb.log({"iteration": iteration,
                  "epoch": epoch, "total loss after refinment": total_loss})
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm_(network_gt.parameters(), grad_clip)
        optimizer.step()


def setup_wandb_run(
    project_name: str,
    run_group: str,
    fold: int,
    lr: float,
    num_epochs: int,
    model_architecture: str,
    num_workers: int,
    batchsize:int,
    seed:int
):
    """
    Sets a new run up (used for k-fold)
    :param str project_name: Name of the project in wandb.
    :param str run_group: Name of the project in wandb.
    :param str fold: number of the executing fold
    :param int lr: learning rate of the model
    :param int num_epochs: number of epochs to train
    :param str model_architecture: Modeltype (architectur) of the model
    :param int num_workers
    :param int batchsize
    :param int seed
    """
    # init wandb
    run = wandb.init(
        settings=wandb.Settings(start_method="thread"),
        project=project_name,
        entity="agroluege",
        name=f"{fold}-Fold",
        group=run_group,
        config={
            "learning rate": lr,
            "epochs": num_epochs,
            "model architecture": model_architecture,
            "num workers": num_workers,
            "batchsize":batchsize,
            "seed": seed
        },
    )
    return run