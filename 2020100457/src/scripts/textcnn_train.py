import os
import sys
import torch
import torch.nn.functional as F


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        device = torch.device("cuda", args.device)
        model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.t_(), target.sub_(1)  # 转置; 减法，0和-1
            if args.cuda:
                device = torch.device("cuda", args.device)
                feature, target = feature.to(device), target.to(device)
            optimizer.zero_grad()
            logits = model(feature)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % 1 == 0:
                corrects = (torch.max(logits, 1)[1].view(
                    target.size()).data == target.data).sum()
                train_acc = 100.0 * corrects / batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.item(),
                                                                             train_acc,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % 100 == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    print(
                        'Saving best model, acc: {:.4f}%\n'.format(best_acc))
                    save(model, args.save_dir, 'best', steps)


def eval(data_iter, model, args, test=False):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.t_(), target.sub_(1)
        if args.cuda:
            device = torch.device("cuda", args.device)
            feature, target = feature.to(device), target.to(device)
        logits = model(feature)
        loss = F.cross_entropy(logits, target)
        avg_loss += loss.item()
        corrects += (torch.max(logits, 1)
                     [1].view(target.size()).data == target.data).sum()
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    if test:
        print('\nTest- loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, accuracy,
                                                                    corrects,
                                                                    size))
    else:
        print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                           accuracy,
                                                                           corrects,
                                                                           size))
    return accuracy


def test(data_iter, model, args):
    model.eval()
    iter_ = iter(data_iter)
    batch = next(iter_)
    feature = batch.text
    feature.t_()
    if args.cuda:
        device = torch.device("cuda", args.device)
        feature = feature.to(device)
    logits = model(feature)
    result = torch.max(logits, 1)[1].data.numpy()[0]

    result = "积极" if result == 1 else "消极"
    print("该句子的情感是:", result)


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)