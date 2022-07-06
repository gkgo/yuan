import os
import tqdm
import time
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.utils.data import DataLoader
# from common.PolyLoss import to_one_hot, PolyLoss

# from common.facolloss import FocalLossV2
from common.meter import Meter, PolyLoss
from common.utils import detect_grad_nan, compute_accuracy, set_seed, setup_run
from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.renet import RENet
from test import test_main, evaluate


def train(epoch, model, loader, optimizer, args=None):
    # (epoch=args.max_epoch,model=RENet(args),loader=train_loaders, optimizer=torch.optim.SGD,)
    model.train()

    train_loader = loader['train_loader']
    train_loader_aux = loader['train_loader_aux']

    # label for query set, always in the same pattern
    label = torch.arange(args.way).repeat(args.query).cuda()  # 012340123401234...

    loss_meter = Meter()  # 创建对象
    acc_meter = Meter()

    k = args.way * args.shot  # 得出带标签图像的个数
    tqdm_gen = tqdm.tqdm(train_loader)

    for i, ((data, train_labels), (data_aux, train_labels_aux)) in enumerate(zip(tqdm_gen, train_loader_aux), 1):  # 这里迭代，调用n_batch次（92）__iter__函数,每个data的batch大小为80
        # data（80，3，84，84） train_labels有80个 data_aux=(64,3,84,84)  train_labels_aux=64个
        data, train_labels = data.cuda(), train_labels.cuda()
        data_aux, train_labels_aux = data_aux.cuda(), train_labels_aux.cuda()

        # Forward images (3, 84, 84) -> (C, H, W)
        model.module.mode = 'encoder'
        data = model(data)
        data_aux = model(data_aux)  # I prefer to separate feed-forwarding data and data_aux due to BN
        # 到这里截至
        # loss for batch
        model.module.mode = 'cca'
        data_shot, data_query = data[:k], data[k:]   # 都是(5,640,5,5)
        logits, absolute_logits = model((data_shot.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1), data_query))
        epi_loss = F.cross_entropy(logits, label)   # Lmetric基于度量的分类损失
        L = PolyLoss()
        train_1 = train_labels[k:]
        train_1 = F.one_hot(train_1, num_classes=64)
        absolute_loss = L(absolute_logits, train_1)
        # absolute_loss = F.cross_entropy(absolute_logits, train_labels[k:])


        # loss for auxiliary batch
        model.module.mode = 'fc'
        logits_aux = model(data_aux)
        train_labels_aux = F.one_hot(train_labels_aux, num_classes=64)
        loss_aux = L(logits_aux, train_labels_aux)
        loss_aux = loss_aux + absolute_loss  # Lanchor损失是分类结果的损失

        loss = args.lamb * epi_loss + loss_aux
        acc = compute_accuracy(logits, label)

        loss_meter.update(loss.item())
        acc_meter.update(acc)
        tqdm_gen.set_description(f'[train] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{acc_meter.avg():.3f} (curr:{acc:.3f})')

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        # detect_grad_nan(model)
        optimizer.step()
        optimizer.zero_grad()

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()


def train_main(args):
    # 数据加载
    Dataset = dataset_builder(args)  # 建立所选择的dataset

    trainset = Dataset('train', args)  # dataset为train.csv
    train_sampler = CategoriesSampler(trainset.label, len(trainset.data) // args.batch, args.way, args.shot + args.query)#gfasdgiasugdfkasgfoigaskgaig
    # （训练数据的标签，n_batch=总数/batch，n_cls=5，n_per=16）train_sampler = {CategoriesSampler: 92}
    # train_sampler是一个自定义的batch_sampler
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=1, pin_memory=True)
    # 每够一个batch，把dataset里的数据按原来顺序，将顺序索引值返回，92组索引

    trainset_aux = Dataset('train', args)
    train_loader_aux = DataLoader(dataset=trainset_aux, batch_size=args.batch, shuffle=True, num_workers=1, pin_memory=True)
    # 每够一个batch，把dataset里的数据打乱顺序，将打乱索引值返回
    train_loaders = {'train_loader': train_loader, 'train_loader_aux': train_loader_aux}  # 不懂
    # 加载验证集
    valset = Dataset('val', args)  # dataset为val.csv
    val_sampler = CategoriesSampler(valset.label, args.val_episode, args.way, args.shot + args.query)  # 返回的n_batch=200

    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=1, pin_memory=True)
    ''' fix val set for all epochs '''
    val_loader = [x for x in val_loader]  # 这里迭代，调用n_batch次（200）__iter__函数,每个batch大小为80

    set_seed(args.seed)
    model = RENet(args).cuda()  # 创建对象model并把数据传输到GPU里(调用renet)
    model = nn.DataParallel(model, device_ids=args.device_ids)  # 如果有多GPU可以在多GPU上运行

    if not args.no_wandb:
        wandb.watch(model)
    print(model)  # 使用wandb可视化工具来输出模型结构

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)  # 定义优化器
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)  # 调整学习率

    max_acc, max_epoch = 0.0, 0
    set_seed(args.seed)

    for epoch in range(1, args.max_epoch + 1):  # 循环
        start_time = time.time()

        train_loss, train_acc, _ = train(epoch, model, train_loaders, optimizer, args)  # 调用train函数
        val_loss, val_acc, _ = evaluate(epoch, model, val_loader, args, set='val')  # 到这步跳出循环，回到main中调用test函数

        if not args.no_wandb:
            wandb.log({'train/loss': train_loss, 'train/acc': train_acc, 'val/loss': val_loss, 'val/acc': val_acc}, step=epoch)
            #  在训练循环中持续记录变化的指标
        if val_acc > max_acc:  # 找出最优正确率，并保存模型
            print(f'[ log ] *********A better model is found ({val_acc:.3f}) *********')
            max_acc, max_epoch = val_acc, epoch
            torch.save(dict(params=model.state_dict(), epoch=epoch), os.path.join(args.save_path, 'max_acc.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_max_acc.pth'))

        if args.save_all:
            torch.save(dict(params=model.state_dict(), epoch=epoch), os.path.join(args.save_path, f'epoch_{epoch}.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, f'optimizer_epoch_{epoch}.pth'))

        epoch_time = time.time() - start_time
        print(f'[ log ] saving @ {args.save_path}')  # 保存模型的路径
        print(f'[ log ] roughly {(args.max_epoch - epoch) / 3600. * epoch_time:.2f} h left\n')  # 大概还要多久跑完

        lr_scheduler.step()  # 更新模型学习率

    return model


if __name__ == '__main__':
    args = setup_run(arg_mode='train')  # 创建对象args

    model = train_main(args)
    test_acc, test_ci = test_main(model, args)

    if not args.no_wandb:
        wandb.log({'test/acc': test_acc, 'test/confidence_interval': test_ci})

