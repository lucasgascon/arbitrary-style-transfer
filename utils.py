def calc_mean_std(x):
    batch_size, num_channels, h, w = x.size()
    x = x.view(batch_size, num_channels, -1)
    mean = x.mean(dim=2).view(batch_size, num_channels, 1, 1)
    std = x.std(dim=2).view(batch_size, num_channels, 1, 1)
    return mean, std

def AdaIN(content, style):
    meanC, stdC = calc_mean_std(content)
    meanS, stdS = calc_mean_std(style)
    return stdS * (content - meanC) / stdC + meanS