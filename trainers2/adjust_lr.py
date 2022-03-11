def adjust_learning_rate(learning_rate, optimizer, epoch):
    lr = learning_rate
    if epoch >= 30:
        lr /= 10
    if epoch >= 40:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr